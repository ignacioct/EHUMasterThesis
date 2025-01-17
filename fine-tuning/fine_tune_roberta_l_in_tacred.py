from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)

# Define TACRED relation slots
TACRED_SLOTS = [
    "per:charges",
    "per:other_family",
    "per:siblings",
    "per:parents",
    "per:children",
    "per:spouse",
    "per:religion",
    "per:employer_or_member_of",
    "per:title",
    "per:schools_attended",
    "per:stateorprovinces_of_residence",
    "per:countries_of_residence",
    "per:cities_of_residence",
    "per:cause_of_death",
    "per:city_of_death",
    "per:stateorprovince_of_death",
    "per:country_of_death",
    "per:date_of_death",
    "per:city_of_birth",
    "per:stateorprovince_of_birth",
    "per:country_of_birth",
    "per:date_of_birth",
    "per:alternate_names",
    "org:website",
    "org:shareholders",
    "org:city_of_headquarters",
    "org:stateorprovince_of_headquarters",
    "org:country_of_headquarters",
    "org:date_dissolved",
    "org:date_founded",
    "org:founded_by",
    "org:parents",
    "org:subsidiaries",
    "org:member_of",
    "org:members",
    "org:number_of_employees_members",
    "org:top_members_employees",
    "org:political_religious_affiliation",
    "org:alternate_names",
    "no_relation",
]


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
        sequence_output = outputs[0]

        head_repr = sequence_output.gather(
            dim=1, index=head_pos.unsqueeze(-1).repeat(1, 1, sequence_output.size(-1))
        ).squeeze(1)
        tail_repr = sequence_output.gather(
            dim=1, index=tail_pos.unsqueeze(-1).repeat(1, 1, sequence_output.size(-1))
        ).squeeze(1)

        entity_pair_repr = torch.cat([head_repr, tail_repr], dim=-1)
        entity_pair_repr = self.dropout(entity_pair_repr)
        entity_pair_repr = torch.tanh(self.dense(entity_pair_repr))
        logits = self.out_proj(self.dropout(entity_pair_repr))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return (
                ((loss,) + (logits,) + outputs[2:])
                if loss is not None
                else (logits,) + outputs[2:]
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Load tokenizer and model configuration
model_name = "FacebookAI/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, add_prefix_space=True
)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])
head_token_id, tail_token_id = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])

config = AutoConfig.from_pretrained(model_name, num_labels=len(TACRED_SLOTS))
model = RobertaForEntityPairClassification.from_pretrained(model_name, config=config)
model.resize_token_embeddings(len(tokenizer))

model.config.label2id = {label: i for i, label in enumerate(TACRED_SLOTS)}
model.config.id2label = {i: label for i, label in enumerate(TACRED_SLOTS)}
model.config.head_token = "[E1]"
model.config.head_token_id = head_token_id
model.config.tail_token = "[E2]"
model.config.tail_token_id = tail_token_id

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def tokenize_function(examples):
    result = tokenizer(examples["text"], padding=False, truncation=True)
    result["head_pos"] = [
        next((i for i, token in enumerate(tokens) if token == head_token_id), -1)
        for tokens in result["input_ids"]
    ]
    result["tail_pos"] = [
        next((i for i, token in enumerate(tokens) if token == tail_token_id), -1)
        for tokens in result["input_ids"]
    ]
    result["label"] = examples["label"]
    return result


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    positive_labels = [
        i for i, label in enumerate(TACRED_SLOTS) if label != "no_relation"
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=positive_labels, average="micro"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def convert_rows_to_training_dict(dataset: pd.DataFrame) -> List:
    output = []
    for _, row in dataset.iterrows():
        if row["relation"] not in TACRED_SLOTS:
            continue
        row["token"].insert(row["subj_start"], "[E1]")
        row["token"].insert(row["subj_end"] + 2, "[/E1]")
        row["token"].insert(row["obj_start"], "[E2]")
        row["token"].insert(row["obj_end"] + 2, "[/E2]")
        text = " ".join(row["token"])
        output.append(
            {"text": text, "head": "T1", "tail": "T2", "label": row["relation"]}
        )
    return output


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, valid_data: pd.DataFrame
) -> DatasetDict:
    train_output = convert_rows_to_training_dict(train_data)
    test_output = convert_rows_to_training_dict(test_data)
    valid_output = convert_rows_to_training_dict(valid_data)
    features = Features(
        {
            "text": Value(dtype="string"),
            "head": Value(dtype="string"),
            "tail": Value(dtype="string"),
            "label": ClassLabel(names=TACRED_SLOTS),
        }
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_output, features=features),
            "test": Dataset.from_list(test_output, features=features),
            "valid": Dataset.from_list(valid_output, features=features),
        }
    )


def train():
    wandb.login()
    with wandb.init(project="TACRED-RoBERTa-Large"):
        train_data = pd.read_json("../data/tacred/train.json")
        test_data = pd.read_json("../data/tacred/test.json")
        valid_data = pd.read_json("../data/tacred/dev.json")

        dataset = preprocess_data(train_data, test_data, valid_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=1e-5,
            weight_decay=0.01,
            num_train_epochs=10,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            output_dir="tmp/",
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()


def main():
    train()


if __name__ == "__main__":
    main()
