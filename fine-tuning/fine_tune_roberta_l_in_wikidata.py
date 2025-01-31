from typing import List

import numpy as np
import pandas as pd
import wandb
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from roberta_for_entity_pair_classification import RobertaForEntityPairClassification
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

WIKIDATA2OUR_SLOTS = {
    "charge": "per:charges",
    "relative": "per:other_family",
    "sibling": "per:siblings",
    "father": "per:parents",
    "mother": "per:parents",
    "child": "per:children",
    "spouse": "per:spouse",
    "religion_or_worldview": "per:religion",
    "employer": "per:employee_or_member_of",
    "noble_title": "per:title",
    "educated_at": "per:schools_attended",
    "residence": "per:place_of_residence",
    "cause_of_death": "per:cause_of_death",
    "place_of_death": "per:place_of_death",
    "date_of_death": "per:date_of_death",
    "place_of_birth": "per:place_of_birth",
    "date_of_birth": "per:date_of_birth",
    "alternative_name": "per:alternative_names",
    "official_website": "org:website",
    "owned_by": "org:shareholders",
    "headquarters_location": "org:headquarters_location",
    "dissolved_abolished_or_demolished": "org:date_dissolved",
    "inception": "org:date_founded",
    "founder": "org:founded_by",
    "parent_organization": "org:parent_organization",
    "subsidiary": "org:subsidiaries",
    "member_of": "org:member_of",
    "number_of_employees": "org:number_of_employees_members",
    "no_relation": "no_relation",
}

# Define Wikidata relation slots as the unique set of values from WIKIDATA2OUR_SLOTS
SLOTS_LIST = list(set(WIKIDATA2OUR_SLOTS.values()))

# Log in to W&B using API key
wandb.login(key="7bd265df21100baa9767bb9f69108bc417db4b4a")

# Load tokenizer and model configuration
model_name = "FacebookAI/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, add_prefix_space=True
)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])
head_token_id, tail_token_id = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])

config = AutoConfig.from_pretrained(model_name, num_labels=len(SLOTS_LIST))
model = RobertaForEntityPairClassification.from_pretrained(model_name, config=config)
model.resize_token_embeddings(len(tokenizer))

model.config.label2id = {label: i for i, label in enumerate(SLOTS_LIST)}
model.config.id2label = {i: label for i, label in enumerate(SLOTS_LIST)}
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
        i for i, label in enumerate(SLOTS_LIST) if label != "no_relation"
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=positive_labels, average="micro"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def convert_rows_to_training_dict(dataset: pd.DataFrame) -> List:
    output = []
    # We need to create columns in the dataset for the start and end positions of the subject and object
    for _, row in dataset.iterrows():
        output.append(
            {
                "text": row["transformed_text"],
                "head": "T1",
                "tail": "T2",
                "label": row["relation"],
            }
        )
    return output


def preprocess_data(train_data: pd.DataFrame, valid_data: pd.DataFrame) -> DatasetDict:
    train_output = convert_rows_to_training_dict(train_data)
    valid_output = convert_rows_to_training_dict(valid_data)
    features = Features(
        {
            "text": Value(dtype="string"),
            "head": Value(dtype="string"),
            "tail": Value(dtype="string"),
            "label": ClassLabel(names=SLOTS_LIST),
        }
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_output, features=features),
            "valid": Dataset.from_list(valid_output, features=features),
        }
    )


def train():
    with wandb.init(project="Wikidata-RoBERTa-Large"):
        train_data = pd.read_csv(
            "../data/wikidata_triplet2text_alpha/wikidata_triplet2text_alpha_generated_train.csv"
        )
        valid_data = pd.read_csv(
            "../data/wikidata_triplet2text_alpha/wikidata_triplet2text_alpha_generated_valid.csv"
        )

        dataset = preprocess_data(train_data, valid_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=0.00003,
            weight_decay=0.01,
            num_train_epochs=5,
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

        trainer.save_model("fine-tuned-roberta-large-tacred")


def main():
    train()


if __name__ == "__main__":
    main()
