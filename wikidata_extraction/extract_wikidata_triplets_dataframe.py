"""
Script for extracting the Wikidata triplets from the Wikidata dump file and saving them in a pandas DataFrame.
"""

import argparse

import pandas as pd
from wikidata_extractor import WikidataExtractor, WikidataExtractorQValues


def main() -> None:
    # Arguments for CLI
    args = argparse.ArgumentParser(
        description="Extract Wikidata triplets from Wikidata dump file."
    )
    args.add_argument(
        "--target",
        type=str,
        help="Either positive or negative triplets.",
        required=True,
    )
    args = args.parse_args()

    if args.target == "positive":
        # Extract triplets with TACRED relations
        extractor = WikidataExtractor(
            "../../data/wikidata/latest-all.json.bz2", "wikidata_triplets.csv"
        )
        # extractor.extract()
        extractor.translate_wikibase_items("q_value_to_label.json")

    elif args.target == "negative":
        # Extract triplets with relations not in TACRED
        df = pd.read_csv("wikidata_triplets.csv")
        q_values = df["q_id"].unique()

        extractor = WikidataExtractorQValues(
            "../../data/wikidata/latest-all.json.bz2", q_values, "wikidata_triplets.csv"
        )
        extractor.extract()
        extractor.translate_wikibase_items()
        extractor.save_to_csv()


if __name__ == "__main__":
    main()
