"""
Script for extracting the Wikidata triplets from the Wikidata dump file and saving them in a pandas DataFrame.
"""

from wikidata_extractor import WikidataExtractor


def main() -> None:
    extractor = WikidataExtractor(
        "../../data/wikidata/latest-all.json.bz2", "wikidata_triplets.csv"
    )
    extractor.extract_with_generator()
    extractor.translate_wikibase_items()
    extractor.save_to_csv()


if __name__ == "__main__":
    main()
