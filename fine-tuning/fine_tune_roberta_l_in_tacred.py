"""
QA Model - Beta model, natural language prompting

This model starts using natural language prompting to ask the LM for the entities.
"""

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


class RobertaForEntityPairClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        head_pos = (
            head_pos.unsqueeze(dim=-1).repeat(1, sequence_output.size(-1)).unsqueeze(1)
        )
        tail_pos = (
            tail_pos.unsqueeze(dim=-1).repeat(1, sequence_output.size(-1)).unsqueeze(1)
        )

        ### Introduce tu código ###

        head_repr = sequence_output.gather(dim=1, index=head_pos).squeeze(1)
        tail_repr = sequence_output.gather(dim=1, index=tail_pos).squeeze(1)

        entity_pair_repr = torch.cat([head_repr, tail_repr], dim=-1)
        entity_pair_repr = self.dropout(entity_pair_repr)
        entity_pair_repr = self.dense(entity_pair_repr)
        entity_pair_repr = torch.tanh(entity_pair_repr)

        #######################

        entity_pair_repr = self.dropout(entity_pair_repr)
        logits = self.out_proj(entity_pair_repr)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# TACRED slots to focus on
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

# Load the model and tokenizer
model_name = "FacebookAI/roberta-large"
project_name = "TACRED-RoBERTa-Large"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, add_prefix_space=True
)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])
head_token_id, tail_token_id = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(TACRED_SLOTS),
)

model = RobertaForEntityPairClassification.from_pretrained(model_name, config=config)

# Defining the Data Collator
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

model.resize_token_embeddings(len(tokenizer))
# Set the correspondences label/ID inside the model config
model.config.label2id = {label: i for i, label in enumerate(TACRED_SLOTS)}
model.config.id2label = {i: label for i, label in enumerate(TACRED_SLOTS)}

model.config.head_token = "[E1]"
model.config.head_token_id = head_token_id
model.config.tail_token = "[E2]"
model.config.tail_token_id = tail_token_id
config = model.config


def tokenize_function(examples):
    # Tokenize the texts
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
    """
        Convert the rows of the dataset to the following format:

        [{
            'text': 'La  [E1] presencia [/E1]  del  [E2] gen [/E2]  de células falciformes y otro normal se denomina
    rasgo drepanocítico.',
            'head': 'T1',
            'tail': 'T2',
            'label': 'subject'
        }]
    """

    output = []

    for _, row in dataset.iterrows():
        if row["relation"] not in TACRED_SLOTS:
            continue

        # Insert in the list of words called token, the markers [E1] and [/E1], which corresponds to the subject entity
        row["token"].insert(row["subj_start"], "[E1]")
        row["token"].insert(row["subj_end"] + 2, "[/E1]")

        # Insert in the list of words called token, the markers [E2] and [/E2], which corresponds to the object entity
        row["token"].insert(row["obj_start"], "[E2]")
        row["token"].insert(row["obj_end"] + 2, "[/E2]")

        # Instead of a string, TACRED has a list of words called token. We need to convert it to a string
        text = " ".join(row["token"])

        output.append(
            {"text": text, "head": "T1", "tail": "T2", "label": row["relation"]}
        )

    return output


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, valid_data: pd.DataFrame
) -> DatasetDict:
    """
        Preprocess the train, test, and validation data, to have the following format:

        [{
            'text': 'La  [E1] presencia [/E1]  del  [E2] gen [/E2]  de células falciformes y otro normal se denomina
    rasgo drepanocítico.',
            'head': 'T1',
            'tail': 'T2',
            'label': 'subject'
        }]

        Return the train, test and validation data in the HuggingFace Dataset format.
    """

    train_data_output = convert_rows_to_training_dict(train_data)
    test_data_output = convert_rows_to_training_dict(test_data)
    valid_data_output = convert_rows_to_training_dict(valid_data)

    label_names = TACRED_SLOTS

    features = Features(
        {
            "text": Value(dtype="string", id=None),
            "head": Value(dtype="string", id=None),
            "tail": Value(dtype="string", id=None),
            "label": ClassLabel(names=label_names, id=None),
        }
    )

    train_dataset = Dataset.from_list(
        train_data_output, features=features, split="train"
    )
    test_dataset = Dataset.from_list(test_data_output, features=features, split="test")
    valid_dataset = Dataset.from_list(
        valid_data_output, features=features, split="valid"
    )

    # Create a dataset dictionary
    dataset_dict = {
        "train": train_dataset,
        "test": test_dataset,
        "valid": valid_dataset,
    }

    return DatasetDict(dataset_dict)


def train(config=None):
    wandb.login(key="7bd265df21100baa9767bb9f69108bc417db4b4a")

    with wandb.init(
        project=project_name,
        config=config,
    ):
        # Loading data
        train_data = pd.read_json("../data/tacred/train.json")
        test_data = pd.read_json("../data/tacred/test.json")
        valid_data = pd.read_json("../data/tacred/dev.json")

        # Preprocess the data
        dataset = preprocess_data(train_data, test_data, valid_data)

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, desc="Tokenizing the dataset"
        )

        # Training
        train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            num_train_epochs=10,
            lr_scheduler_type="constant",
            seed=42,
            fp16=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="f1",
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

    train_result = trainer.train()
    metrics = train_result.metrics
    print(metrics)


def main():
    train()


if __name__ == "__main__":
    main()
