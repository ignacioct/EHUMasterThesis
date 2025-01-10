import pandas as pd
from jinja2 import Template
from tqdm import tqdm
from vllm import LLM, SamplingParams


def main():
    # Define the LLM model
    model = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Define the sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=100, stop=None
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
        with open("alpha_gen_template_advanced.jinja2") as file_:
            template = Template(file_.read())
        rendered_prompt = template.render()

        user_prompt = f"Here is the triplet:\nTriplet: [{row['q_name']} | {row['p_name']} | {row['p_value']}]\n\n Please generate a sentence that describes the triplet."

        chat_input = [
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = model.chat(chat_input, sampling_params, use_tqdm=False)
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
            },
            index=[0],
        )

        # Concatenate the new row to output_df
        output = pd.concat([output, new_row], ignore_index=True)

    # Save the output dataframe
    output.to_csv("wikidata_triplet2text_alpha_generated.csv", index=False)


if __name__ == "__main__":
    main()
