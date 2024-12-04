"""
Script for extracting the Wikidata triplets from the Wikidata dump file and saving them in a pandas DataFrame.
"""

import pandas as pd
from wikidata_extractor import WikidataExtractorQValues


def main() -> None:
    # Extract triplets with TACRED relations
    # extractor = WikidataExtractor(
    #     "../../data/wikidata/latest-all.json.bz2", "wikidata_triplets.csv"
    # )
    # extractor.extract_with_generator()
    # extractor.translate_wikibase_items()
    # extractor.save_to_csv()

    # Extract triplets with relations not in TACRED

    df = pd.read_csv("wikidata_triplets.csv")
    q_values = df["q_id"].unique()

    extractor = WikidataExtractorQValues(
        "../../data/wikidata/latest-all.json.bz2", q_values, "wikidata_triplets.csv"
    )
    extractor.extract_with_generator()
    extractor.translate_wikibase_items()
    extractor.save_to_csv()


if __name__ == "__main__":
    main()
