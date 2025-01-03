import pandas as pd
from jinja2 import Template
from tqdm import tqdm
from vllm import LLM, SamplingParams


def main():
    # Define the LLM model
    model = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Define the sampling parameters
    sampling_params = SamplingParams(
        temperature=0.5,
    )

    # Load the dataset
    dataset = pd.read_csv("wikidata_triplet2text_alpha.csv")

    # Set up the output dataframe
    output = pd.DataFrame(
        columns=[
            "q_id",
            "q_name",
            "p_id",
            "p_name",
            "p_value",
            "p_value_type",
            "positive_negative",
            "text",
        ]
    )

    for _, row in tqdm(
        dataset.iterrows(), total=len(dataset), desc="Generating text..."
    ):
        with open("alpha_gen_template.jinja2") as file_:
            template = Template(file_.read())
        rendered_prompt = template.render(
            subject=row["q_name"], relation=row["p_name"], object=row["p_value"]
        )

        outputs = model.generate([rendered_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Create a new row with the original data and generated text
        new_row = pd.DataFrame(
            {
                "q_id": row["q_id"],
                "q_name": row["q_name"],
                "p_id": row["p_id"],
                "p_name": row["p_name"],
                "p_value": row["p_value"],
                "p_value_type": row["p_value_type"],
                "positive_negative": row["positive_negative"],
                "text": generated_text,
            }
        )

        # Concatenate the new row to output_df
        output = pd.concat([output, new_row], ignore_index=True)

    # Save the output dataframe
    output.to_csv("wikidata_triplet2text_alpha_generated.csv", index=False)


if __name__ == "__main__":
    main()
