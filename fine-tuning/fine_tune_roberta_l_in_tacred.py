import ast
from typing import List, Tuple

import numpy as np
import pandas as pd
import wandb
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from roberta_for_entity_pair_classification import RobertaForEntityPairClassification
from sklearn.metrics import precision_recall_fscore_support
from slots_definition import SLOTS_LIST
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(p: Tuple) -> dict:
    """
    Compute metrics for the model, so there is information about precision, recall and f1 during the evaluation.

    Args:
        p(Tuple): The predictions from the model.

    Returns:
        dict: The metrics for the model.
    """

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
    """
    Convert the rows of the dataset to a list of dictionaries that can be used to create a Hugging Face dataset.

    Args:
        dataset(pd.DataFrame): The dataset to convert.

    Returns:
        List: The list of dictionaries that can be used to create a Hugging Face dataset.
    """

    output = []
    for _, row in dataset.iterrows():
        if row["relation"] not in SLOTS_LIST:
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
    """
    Preprocess the data to be used in the model.

    Args:
        train_data(pd.DataFrame): The training data.
        valid_data(pd.DataFrame): The validation data.

    Returns:
        DatasetDict: The preprocessed data.
    """

    # The token column comes as a string, but needs to be converted to list
    train_data["token"] = train_data["token"].apply(ast.literal_eval)
    test_data["token"] = test_data["token"].apply(ast.literal_eval)
    valid_data["token"] = valid_data["token"].apply(ast.literal_eval)

    train_output = convert_rows_to_training_dict(train_data)
    test_output = convert_rows_to_training_dict(test_data)
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
            "test": Dataset.from_list(test_output, features=features),
            "valid": Dataset.from_list(valid_output, features=features),
        }
    )


def train() -> None:
    """
    Train the model on the Wikidata dataset.
    """

    with wandb.init(project="TACRED-RoBERTa-Large"):
        # Load & preprocess datasets
        train_data = pd.read_csv("../data/tacred/train.csv")
        test_data = pd.read_csv("../data/tacred/test.csv")
        valid_data = pd.read_csv("../data/tacred/valid.csv")
        dataset = preprocess_data(train_data, test_data, valid_data)

        # Load the tokenizer & tokenize the dataset
        model_name = "FacebookAI/roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, add_prefix_space=True
        )
        tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])
        head_token_id, tail_token_id = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])

        def tokenize_function(examples: dict) -> dict:
            """
            Tokenize the examples.
            Nested function, so we don't need to define the tokenizer and head/tail token outside the train function.

            Args:
                examples(dict): The examples to tokenize.

            Returns:
                dict: The tokenized examples.
            """
            result = tokenizer(examples["text"], padding=False, truncation=True)
            result["head_pos"] = [
                next(
                    (i for i, token in enumerate(tokens) if token == head_token_id), -1
                )
                for tokens in result["input_ids"]
            ]
            result["tail_pos"] = [
                next(
                    (i for i, token in enumerate(tokens) if token == tail_token_id), -1
                )
                for tokens in result["input_ids"]
            ]
            result["label"] = examples["label"]
            return result

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Load the model
        config = AutoConfig.from_pretrained(model_name, num_labels=len(SLOTS_LIST))
        model = RobertaForEntityPairClassification.from_pretrained(
            model_name, config=config
        )

        # Model configuration
        model.resize_token_embeddings(len(tokenizer))
        model.config.label2id = {label: i for i, label in enumerate(SLOTS_LIST)}
        model.config.id2label = {i: label for i, label in enumerate(SLOTS_LIST)}
        model.config.head_token = "[E1]"
        model.config.head_token_id = head_token_id
        model.config.tail_token = "[E2]"
        model.config.tail_token_id = tail_token_id

        # Load the data collator
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        # Training arguments
        train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=5,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            output_dir="tmp/",
            warmup_ratio=0.1,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model("fine-tuned-roberta-large-tacred")


def main():
    # Log in to W&B using API key
    wandb.login(key="7bd265df21100baa9767bb9f69108bc417db4b4a")

    train()


if __name__ == "__main__":
    main()
