"""
Script for the first generation of synthetic text from the positive triplets of the Wikidata dataset. Here, we obtain:
- One sentence per positive triplet
"""

import pandas as pd
import torch
import transformers
from tqdm import tqdm


def main() -> None:
    # Load the input and create the output dataset
    triplets_df = pd.read_csv("wikidata_triplets.csv")
    output_df = pd.DataFrame(
        columns=[
            "q_id",
            "q_name",
            "p_id",
            "p_name",
            "p_value",
            "p_value_type",
            "synthetic_text",
        ]
    )

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    system_prompt = """
    You are a AI assistant aimed to generate a sentences given a triplet with a subject, a relation and the object of a relation.
    Please generate a sentence that describes the triplet. If there is a date and time, please ignore the time.
    Be verbose, and use rich and elaborate sentences.
    """

    for _, row in tqdm(
        triplets_df.iterrows(),
        total=triplets_df.shape[0],
        desc="Generating synthetic text",
    ):
        # Integrate the system and the user prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"\n\nHere is the triplet: {row["q_name"]}, {row["p_name"]}, {row["p_value"]}",
            },
        ]

        # Execute the pipeline
        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )

        # Extend the output dataframe
        new_row = {
            "q_id": row["q_id"],
            "q_name": row["q_name"],
            "p_id": row["p_id"],
            "p_name": row["p_name"],
            "p_value": row["p_value"],
            "p_value_type": row["p_value_type"],
            "synthetic_text": outputs[0]["generated_text"][-1],
        }
        output_df = output_df.append(new_row, ignore_index=True)

    # Save the output dataframe
    output_df.to_csv("wikidata_triplets_positive.csv", index=False)


if __name__ == "__main__":
    main()
