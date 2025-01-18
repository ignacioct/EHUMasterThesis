from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


# Custom model for entity pair classification
class RobertaForEntityPairClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout or config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        head_pos: Optional[torch.LongTensor] = None,
        tail_pos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get the base model outputs
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[
            0
        ]  # Shape: (batch_size, sequence_length, hidden_size)
        batch_size = sequence_output.size(0)

        # Ensure head_pos and tail_pos are the right shape
        head_pos = head_pos.view(batch_size, 1, 1).expand(
            -1, -1, sequence_output.size(-1)
        )
        tail_pos = tail_pos.view(batch_size, 1, 1).expand(
            -1, -1, sequence_output.size(-1)
        )

        # Extract entity representations
        head_repr = torch.gather(sequence_output, 1, head_pos).squeeze(1)
        tail_repr = torch.gather(sequence_output, 1, tail_pos).squeeze(1)

        # Combine entity representations
        entity_pair_repr = torch.cat([head_repr, tail_repr], dim=-1)
        entity_pair_repr = self.dropout(entity_pair_repr)
        entity_pair_repr = torch.tanh(self.dense(entity_pair_repr))

        # Get logits - ensure consistent shape ADDED BY IGNACIO
        logits = self.out_proj(
            self.dropout(entity_pair_repr)
        )  # Shape: (batch_size, num_labels)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
