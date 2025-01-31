from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from roberta_for_entity_pair_classification import RobertaForEntityPairClassification
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from slots_definition import SLOTS_LIST
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

# Load your saved model and tokenizer
model = RobertaForEntityPairClassification.from_pretrained(
    "fine-tuned-roberta-large-wikidata"
)
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-roberta-large-wikidata")
head_token_id, tail_token_id = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])


def evaluate_model(trainer: Trainer, test_dataset) -> Tuple[Dict, np.ndarray, List]:
    """
    Evaluate model on test dataset and return metrics and predictions
    """
    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Convert logits to predicted classes
    pred_classes = np.argmax(logits, axis=-1)

    # Get label names for better readability
    label_names = [trainer.model.config.id2label[i] for i in range(len(SLOTS_LIST))]

    # Calculate metrics for all relations
    metrics = {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average=None, labels=range(len(SLOTS_LIST))
    )

    # Store per-class metrics
    for i, label in enumerate(label_names):
        metrics[f"{label}_precision"] = precision[i]
        metrics[f"{label}_recall"] = recall[i]
        metrics[f"{label}_f1"] = f1[i]

    # Calculate micro averages excluding no_relation
    positive_labels = [
        i for i, label in enumerate(label_names) if label != "no_relation"
    ]
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, pred_classes, labels=positive_labels, average="micro"
    )

    metrics.update(
        {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }
    )

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, pred_classes, labels=range(len(SLOTS_LIST)))

    return metrics, conf_matrix, label_names


def plot_confusion_matrix(conf_matrix: np.ndarray, labels: List[str], output_path: str):
    """
    Plot and save confusion matrix heatmap
    """
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        conf_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def convert_rows_to_training_dict_wikidata(dataset: pd.DataFrame) -> List:
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


def preprocess_data_wikidata(test_data: pd.DataFrame) -> DatasetDict:
    test_output = convert_rows_to_training_dict_wikidata(test_data)
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
            "test": Dataset.from_list(test_output, features=features),
        }
    )


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


def evaluate_wikidata(df: Dataset, output_str) -> str:
    # Evaluate on Wikidata test data
    dataset = preprocess_data_wikidata(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluate
    metrics, conf_matrix, label_names = evaluate_model(
        trainer, tokenized_dataset["test"]
    )

    # Print metrics
    output_str += "\nEvaluation over Wikidata test set:"
    output_str += f"\nMicro Precision: {metrics['micro_precision']:.4f}"
    output_str += f"\nMicro Recall: {metrics['micro_recall']:.4f}"
    output_str += f"\nMicro F1: {metrics['micro_f1']:.4f}"

    output_str += "\nPer-relation metrics:"
    for label in label_names:
        if label != "no_relation":
            output_str += f"\n{label}:"
            output_str += f"\nPrecision: {metrics[f'{label}_precision']:.4f}"
            output_str += f"\nRecall: {metrics[f'{label}_recall']:.4f}"
            output_str += f"\nF1: {metrics[f'{label}_f1']:.4f}"

    output_str += "\n\n\n"
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, label_names, "confusion_matrix_wikidata.png")

    return output_str


def main():
    # Create a string representation of the output
    output_str = ""

    # Load and preprocess test data
    test_data_wikidata = pd.read_csv(
        "../data/wikidata_triplet2text_alpha/wikidata_triplet2text_alpha_generated_test.csv"
    )
    # test_data_tacred = pd.read_csv("../data/tacred/test.csv")

    output_str = evaluate_wikidata(test_data_wikidata, output_str)

    # Print the output string in a file
    with open("results_test_wikidata.txt", "w") as f:
        f.write(output_str)


if __name__ == "__main__":
    main()
