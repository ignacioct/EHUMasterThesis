import bz2
import json
import os
from datetime import datetime
from typing import List, Union

import pandas as pd
from tqdm import tqdm


def process_symbolic_link(symlink_path: str, subdirectory_level_1: bool = True) -> str:
    """
    Process a symbolic link and return its real path.

    This function takes a symbolic link path, resolves it to its absolute path,
    and then returns the real path that the symlink points to. If the input path
    is not a symlink, it returns the absolute path.

    Args:
        symlink_path (str): The path to the symbolic link or file.
        subdirectory_level_1 (bool, optional): If True, prepend "../" to the real path.
            Defaults to True.

    Returns:
        str: The real path that the symlink points to, or the absolute path if
            the input is not a symlink. If subdirectory_level_1 is True, "../"
            is prepended to the path.
    """

    if os.path.islink(symlink_path):
        real_path = os.readlink(symlink_path)
    else:
        real_path = os.path.abspath(symlink_path)

    if subdirectory_level_1:
        return os.path.join("../..", real_path)

    return real_path


class WikidataExtractor:
    def __init__(self, wikidata_dump_path: str, output_path: Union[str, None] = None):
        self.wikidata_dump_path = process_symbolic_link(wikidata_dump_path)

        # Define the slots for the person and organization entities
        self.slots_name2id = {
            "charge": "P1595",
            "relative": "P1038",
            "sibling": "P3373",
            "father": "P22",
            "mother": "P25",
            "child": "P40",
            "spouse": "P26",
            "religion_or_worldview": "P140",
            "employer": "P108",
            "member_of": "P463",
            "noble_title": "P97",
            "educated_at": "P69",
            "residence": "P551",
            "country_of_citizenship": "P27",
            "cause_of_death": "P509",
            "place_of_death": "P20",
            "date_of_death": "P570",
            "place_of_birth": "P19",
            "date_of_birth": "P569",
            "alternative_name": "P4970",
            "official_website": "P856",
            "owned_by": "P127",
            "headquarters_location": "P159",
            "dissolved_abolished_or_demolished": "P576",
            "inception": "P571",
            "founder": "P112",
            "parent_organization": "P749",
            "subsidiary": "P355",
            "number_of_employees": "P1128",
            "number_of_members": "P1129",
            "instance_of": "P31",
        }

        self.slots_id2name = {v: k for k, v in self.slots_name2id.items()}

        self.slots_id_list = list(self.slots_name2id.values())

        self.output_path = output_path

    def extract(
        self,
        total_limit: int = 20000,
    ) -> pd.DataFrame:
        """
        Extract the relevant data from the Wikidata dump file.

        This function reads the Wikidata dump file line by line, parses the JSON data,
        and extracts the relevant information for the specified properties. It accumulates
        the extracted data into a pandas DataFrame and returns it.

        Args:
            total_limit (int, optional): The number of unique Q-values to process.

        Returns:
            str: The path to the output .csv file.
        """

        # Initialize the output DataFrame with predefined columns
        output_df = pd.DataFrame(
            columns=["q_id", "q_name", "p_id", "p_name", "p_value", "p_value_type"]
        )

        counter_person = 0
        counter_organization = 0
        limit_person = total_limit // 2
        limit_organization = total_limit // 2

        # Create an output .csv file to append each row in each iteration
        if self.output_path:
            output_df.to_csv(self.output_path, index=False)

        # Open the compressed file
        with bz2.open(self.wikidata_dump_path, mode="rb") as f:
            # Use tqdm for a progress bar
            with tqdm(desc="Processing lines", unit="line", total=total_limit) as pbar:
                for i, line in enumerate(f):
                    try:
                        # Decode the line and clean unnecessary characters
                        line_str = line.decode("utf-8").rstrip(",\n")

                        # Skip irrelevant or empty lines
                        if line_str in ["[", "]"] or len(line_str) == 0:
                            continue

                        # Parse the line as JSON
                        item = json.loads(line_str)

                        # Process only "item" type entities
                        if item.get("type") == "item":
                            q_id = item["id"]

                            # Attempt to get the English label; skip if not available
                            q_name = item.get("labels", {}).get("en", {}).get("value")
                            if not q_name:
                                continue

                            output_rows = []
                            is_instance_of_person = False
                            is_instance_of_organization = False

                            # Iterate through the item's claims (properties)
                            for prop_id, prop_values in item.get("claims", {}).items():
                                # Process only properties of interest
                                if prop_id not in self.slots_id_list:
                                    continue

                                try:
                                    # Extract property details
                                    p_id = prop_id
                                    p_name = self.slots_id2name.get(p_id, "Unknown")

                                    # Get the first property's value and its datatype
                                    p_value_type = prop_values[0]["mainsnak"][
                                        "datatype"
                                    ]
                                    p_value = self.process_p_value(
                                        prop_values, p_value_type
                                    )

                                    # Append the data as a new row in the DataFrame
                                    output_rows.append(
                                        {
                                            "q_id": q_id,
                                            "q_name": q_name,
                                            "p_id": prop_id,
                                            "p_name": p_name,
                                            "p_value": p_value,
                                            "p_value_type": p_value_type,
                                        }
                                    )

                                    # Check if the entity is a person or organization
                                    if p_id == self.slots_name2id["instance_of"]:
                                        if p_value == "Q43229":
                                            is_instance_of_organization = True
                                        elif p_value == "Q5":
                                            # Avoid use Q215627 for P31, use Q5 instead
                                            # https://www.wikidata.org/wiki/Q215627
                                            is_instance_of_person = True

                                except KeyError:
                                    # Handle missing expected keys in the property data
                                    continue

                                except ValueError:
                                    continue

                            # We only want to extract the data for the person and organization entities
                            if is_instance_of_person and counter_person < limit_person:
                                # Append the rows to the output .csv file
                                if self.output_path:
                                    pd.DataFrame(output_rows).to_csv(
                                        self.output_path,
                                        mode="a",
                                        header=False,
                                        index=False,
                                    )

                                counter_person += 1  # Increment the counter
                                pbar.update(1)  # Update the progress bar

                            elif (
                                is_instance_of_organization
                                and counter_organization < limit_organization
                            ):
                                # Append the rows to the output .csv file
                                if self.output_path:
                                    pd.DataFrame(output_rows).to_csv(
                                        self.output_path,
                                        mode="a",
                                        header=False,
                                        index=False,
                                    )

                                counter_organization += 1
                                pbar.update(1)  # Update the progress bar

                            if (
                                counter_person >= limit_person
                                and counter_organization >= limit_organization
                            ):
                                break

                    except json.JSONDecodeError as json_err:
                        print(f"JSON decoding error at line {i}: {json_err}")
                        continue
                    except UnicodeDecodeError as unicode_err:
                        print(f"Unicode decoding error at line {i}: {unicode_err}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error at line {i}: {e}")
                        print(f"Error happened at line {i} with content: {line_str}")
                        print(f"Error type: {type(e)}")
                        break

        return self.output_path

    def process_p_value(self, prop_values: dict, p_value_type: str) -> str:
        """
        Process the property value based on its datatype.

        This function processes the property value based on its datatype and
        returns the processed value as a string.

        Args:
            prop_values (dict): The property values from the Wikidata dump.
            p_value_type (str): The datatype of the property value.

        Returns:
            str: The processed property value as a string.
        """

        if p_value_type == "quantity":
            return self.process_quantity(prop_values)
        elif p_value_type == "string":
            return self.process_string(prop_values)
        elif p_value_type == "time":
            return self.process_time(prop_values)
        elif p_value_type == "url":
            return self.process_url(prop_values)
        elif p_value_type == "wikibase-item":
            return self.process_wikibase_item(prop_values)
        else:
            raise ValueError(f"Unsupported property value type: {p_value_type}")

    def process_quantity(self, prop_values: dict) -> float:
        """
        Process the quantity value from the Wikidata dump.

        This function processes the quantity value from the Wikidata dump, which
        includes the amount and the unit. It returns the amount as a float value.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            float: The quantity amount as a float value.
        """

        quantity = prop_values[0]["mainsnak"]["datavalue"]["value"]["amount"]

        # If quantity starts by +, remove it
        if quantity.startswith("+"):
            quantity = quantity[1:]

        return float(quantity)

    def process_string(self, prop_values: dict) -> str:
        """
        Process the string value from the Wikidata dump.

        This function processes the string value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The string value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]

    def process_time(self, prop_values: dict) -> str:
        """
        Process the time value from the Wikidata dump.

        This function processes the time value from the Wikidata dump, which
        includes the timestamp and timezone offset. It returns the timestamp
        as a human-readable string.

        Example of wikipedia time format: "+2021-08-01T00:00:00Z"

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The human-readable timestamp string.
        """

        time_dict = prop_values[0]["mainsnak"]["datavalue"]["value"]

        iso_time = time_dict["time"]

        # Remove the leading '+' if present
        if iso_time.startswith("+"):
            iso_time = iso_time[1:]

        # Parse the ISO 8601 timestamp
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))

        # Format the date and time into a readable string
        readable_time = dt.strftime("%B %d, %Y at %I:%M:%S %p")

        # Add CE designation (since we know it's not BCE due to the original '+')
        readable_time += " CE"

        # Add timezone information
        timezone = time_dict["timezone"]
        if timezone == 0:
            tz_info = "UTC"
        else:
            tz_info = f"UTC{'+' if timezone >= 0 else ''}{timezone}"

        return f"{readable_time} {tz_info}"

    def process_url(self, prop_values: dict) -> str:
        """
        Process the URL value from the Wikidata dump.

        This function processes the URL value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The URL value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]

    def process_wikibase_item(self, prop_values: dict) -> str:
        """
        Process the Wikibase item value from the Wikidata dump.

        This function processes the Wikibase item value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The Wikibase item value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]["id"]

    def translate_wikibase_items(self, json_path: Union[str, None]) -> pd.DataFrame:
        """
        Translate Wikibase items to their English labels.

        This function translates Wikibase items to their English labels using a dictionary
        created from the Wikidata dump file. It updates the property values in the DataFrame
        and returns the updated DataFrame.

        Args:
            json_path (str, None): The path to the JSON file containing the q_value_to_label dictionary.

        Returns:
            str: The path to the output .csv file.
        """

        if json_path is None:
            # Create a dictionary to translate Wikibase items
            q_value_to_label = self.get_q_value_to_label()

            # Output q_value_to_label dictionary to a file
            with open("q_value_to_label.json", "w") as f:
                print("Saving q_value_to_label dictionary to q_value_to_label.json")
                json.dump(q_value_to_label, f)
        else:
            # Load the q_value_to_label dictionary from the JSON file
            with open(json_path, "r") as f:
                q_value_to_label = json.load(f)

        # Open the output .csv file
        wikidata_triplets_df = pd.read_csv(self.output_path)

        for index, row in tqdm(
            wikidata_triplets_df.iterrows(),
            total=len(wikidata_triplets_df),
            desc="Translating Wikibase items",
            unit="row",
        ):
            if row["p_value_type"] == "wikibase-item":
                q_id = row["p_value"]
                q_label = q_value_to_label.get(q_id)
                if q_label:
                    wikidata_triplets_df.at[index, "p_value"] = q_label

        # Save the updated DataFrame to the output .csv file
        wikidata_triplets_df.to_csv(self.output_path, index=False)

        return self.output_path

    def get_q_value_to_label(self) -> dict:
        """
        Create a dictionary to translate Wikibase items.

        This function reads the Wikidata dump file line by line, parses the JSON data,
        and creates a dictionary to translate Wikibase items.

        Returns:
            dict: The dictionary to translate Wikibase items.
        """

        q_value_to_label = {}

        with bz2.open(self.wikidata_dump_path, mode="rb") as f:
            with tqdm(
                desc="Creating the dictionary to translate wikibase items", unit="line"
            ) as pbar:
                for _, line in enumerate(f):
                    try:
                        line_str = line.decode("utf-8").rstrip(",\n")
                        if line_str in ["[", "]"] or len(line_str) == 0:
                            continue

                        item = json.loads(line_str)
                        if item.get("type") == "item":
                            q_id = item["id"]
                            q_label = item.get("labels", {}).get("en", {}).get("value")
                            if q_label:
                                q_value_to_label[q_id] = q_label

                        pbar.update(1)

                    except KeyError:
                        # Handle missing expected keys in the property data
                        continue

                    except ValueError:
                        continue

        return q_value_to_label


class WikidataExtractorQValues:
    def __init__(
        self,
        wikidata_dump_path: str,
        focus_q_values: List[str],
        output_path: Union[str, None] = None,
    ):
        self.wikidata_dump_path = process_symbolic_link(wikidata_dump_path)

        # Define the slots for the person and organization entities
        self.slots_name2id = {
            "charge": "P1595",
            "relative": "P1038",
            "sibling": "P3373",
            "father": "P22",
            "mother": "P25",
            "child": "P40",
            "spouse": "P26",
            "religion_or_worldview": "P140",
            "employer": "P108",
            "member_of": "P463",
            "noble_title": "P97",
            "educated_at": "P69",
            "residence": "P551",
            "country_of_citizenship": "P27",
            "cause_of_death": "P509",
            "place_of_death": "P20",
            "date_of_death": "P570",
            "place_of_birth": "P19",
            "date_of_birth": "P569",
            "alternative_name": "P4970",
            "official_website": "P856",
            "owned_by": "P127",
            "headquarters_location": "P159",
            "dissolved_abolished_or_demolished": "P576",
            "inception": "P571",
            "founder": "P112",
            "parent_organization": "P749",
            "subsidiary": "P355",
            "number_of_employees": "P1128",
            "number_of_members": "P1129",
        }

        self.focus_q_values = focus_q_values

        self.slots_id2name = {v: k for k, v in self.slots_name2id.items()}

        self.slots_id_list = list(self.slots_name2id.values())

        self.wikidata_triplets_df = pd.DataFrame(
            columns=["q_id", "q_name", "p_id", "p_name", "p_value", "p_value_type"]
        )

        self.output_path = output_path

    def extract(
        self,
        total_limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Extract the relevant data from the Wikidata dump file.

        This function reads the Wikidata dump file line by line, parses the JSON data,
        and extracts the relevant information for the specified properties. It accumulates
        the extracted data into a pandas DataFrame and returns it.

        Args:
            total_limit (int, optional): The number of lines to process.

        Returns:
            pd.DataFrame: The extracted data as a pandas DataFrame.
        """

        # Initialize the output DataFrame with predefined columns
        output_df = pd.DataFrame(
            columns=["q_id", "q_name", "p_id", "p_name", "p_value", "p_value_type"]
        )

        counter = 0

        # Create an output .csv file to append each row in each iteration
        if self.output_path:
            output_df.to_csv(self.output_path, index=False)

        # Open the compressed file
        with bz2.open(self.wikidata_dump_path, mode="rb") as f:
            # Use tqdm for a progress bar
            with tqdm(desc="Processing lines", unit="line", total=total_limit) as pbar:
                for i, line in enumerate(f):
                    # Stop if we reach the total limit
                    if counter >= total_limit:
                        break

                    try:
                        # Decode the line and clean unnecessary characters
                        line_str = line.decode("utf-8").rstrip(",\n")

                        # Skip irrelevant or empty lines
                        if line_str in ["[", "]"] or len(line_str) == 0:
                            continue

                        # Parse the line as JSON
                        item = json.loads(line_str)

                        if item.get("type") == "item":
                            q_id = item["id"]
                            q_name = item.get("labels", {}).get("en", {}).get("value")

                            # We only want to extract the data for the focus Q-values
                            if not q_name or q_id not in self.focus_q_values:
                                continue

                        # Process only "item" type entities
                        if item.get("type") == "item":
                            q_id = item["id"]

                            # Attempt to get the English label; skip if not available
                            q_name = item.get("labels", {}).get("en", {}).get("value")
                            if not q_name:
                                continue

                            # Iterate through the item's claims (properties)
                            for prop_id, prop_values in item.get("claims", {}).items():
                                # Process only properties of interest
                                if prop_id not in self.slots_id_list:
                                    continue

                                try:
                                    # Extract property details
                                    p_id = prop_id
                                    p_name = self.slots_id2name.get(p_id, "Unknown")

                                    # Get the first property's value and its datatype
                                    p_value_type = prop_values[0]["mainsnak"][
                                        "datatype"
                                    ]
                                    p_value = self.process_p_value(
                                        prop_values, p_value_type
                                    )

                                    # Append the data as a new row in the DataFrame
                                    output_row = {
                                        "q_id": q_id,
                                        "q_name": q_name,
                                        "p_id": prop_id,
                                        "p_name": p_name,
                                        "p_value": p_value,
                                        "p_value_type": p_value_type,
                                    }

                                    # Append the row to the output .csv file
                                    if self.output_path:
                                        pd.DataFrame([output_row]).to_csv(
                                            self.output_path,
                                            mode="a",
                                            header=False,
                                            index=False,
                                        )

                                    counter += 1  # Increment the counter

                                    pbar.update(1)  # Update the progress bar

                                    # Also stop if we reach the total limit
                                    if counter >= total_limit:
                                        break

                                except KeyError:
                                    # Handle missing expected keys in the property data
                                    continue

                                except ValueError:
                                    continue

                    except json.JSONDecodeError as json_err:
                        print(f"JSON decoding error at line {i}: {json_err}")
                        continue
                    except UnicodeDecodeError as unicode_err:
                        print(f"Unicode decoding error at line {i}: {unicode_err}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error at line {i}: {e}")
                        print(f"Error happened at line {i} with content: {line_str}")
                        print(f"Error type: {type(e)}")
                        break

        self.wikidata_triplets_df = pd.read_csv(self.output_path)
        return self.wikidata_triplets_df

    def process_p_value(self, prop_values: dict, p_value_type: str) -> str:
        """
        Process the property value based on its datatype.

        This function processes the property value based on its datatype and
        returns the processed value as a string.

        Args:
            prop_values (dict): The property values from the Wikidata dump.
            p_value_type (str): The datatype of the property value.

        Returns:
            str: The processed property value as a string.
        """

        if p_value_type == "quantity":
            return self.process_quantity(prop_values)
        elif p_value_type == "string":
            return self.process_string(prop_values)
        elif p_value_type == "time":
            return self.process_time(prop_values)
        elif p_value_type == "url":
            return self.process_url(prop_values)
        elif p_value_type == "wikibase-item":
            return self.process_wikibase_item(prop_values)
        else:
            raise ValueError(f"Unsupported property value type: {p_value_type}")

    def process_quantity(self, prop_values: dict) -> float:
        """
        Process the quantity value from the Wikidata dump.

        This function processes the quantity value from the Wikidata dump, which
        includes the amount and the unit. It returns the amount as a float value.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            float: The quantity amount as a float value.
        """

        quantity = prop_values[0]["mainsnak"]["datavalue"]["value"]["amount"]

        # If quantity starts by +, remove it
        if quantity.startswith("+"):
            quantity = quantity[1:]

        return float(quantity)

    def process_string(self, prop_values: dict) -> str:
        """
        Process the string value from the Wikidata dump.

        This function processes the string value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The string value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]

    def process_time(self, prop_values: dict) -> str:
        """
        Process the time value from the Wikidata dump.

        This function processes the time value from the Wikidata dump, which
        includes the timestamp and timezone offset. It returns the timestamp
        as a human-readable string.

        Example of wikipedia time format: "+2021-08-01T00:00:00Z"

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The human-readable timestamp string.
        """

        time_dict = prop_values[0]["mainsnak"]["datavalue"]["value"]

        iso_time = time_dict["time"]

        # Remove the leading '+' if present
        if iso_time.startswith("+"):
            iso_time = iso_time[1:]

        # Parse the ISO 8601 timestamp
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))

        # Format the date and time into a readable string
        readable_time = dt.strftime("%B %d, %Y at %I:%M:%S %p")

        # Add CE designation (since we know it's not BCE due to the original '+')
        readable_time += " CE"

        # Add timezone information
        timezone = time_dict["timezone"]
        if timezone == 0:
            tz_info = "UTC"
        else:
            tz_info = f"UTC{'+' if timezone >= 0 else ''}{timezone}"

        return f"{readable_time} {tz_info}"

    def process_url(self, prop_values: dict) -> str:
        """
        Process the URL value from the Wikidata dump.

        This function processes the URL value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The URL value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]

    def process_wikibase_item(self, prop_values: dict) -> str:
        """
        Process the Wikibase item value from the Wikidata dump.

        This function processes the Wikibase item value from the Wikidata dump.

        Args:
            prop_values (dict): The property values from the Wikidata dump.

        Returns:
            str: The Wikibase item value.
        """

        return prop_values[0]["mainsnak"]["datavalue"]["value"]["id"]

    def translate_wikibase_items(self) -> pd.DataFrame:
        """
        Translate Wikibase items to their English labels.

        This function translates Wikibase items to their English labels using a dictionary
        created from the Wikidata dump file. It updates the property values in the DataFrame
        and returns the updated DataFrame.

        Returns:
            pd.DataFrame: The updated DataFrame with the Wikibase items translated to English labels.

        """

        q_value_to_label, p_value_to_label = self.get_q_p_value_to_label()

        for index, row in tqdm(
            self.wikidata_triplets_df.iterrows(),
            total=len(self.wikidata_triplets_df),
            desc="Translating Wikibase items",
            unit="row",
        ):
            if row["p_value_type"] == "wikibase-item":
                q_id = row["p_value"]
                q_label = q_value_to_label.get(q_id)
                if q_label:
                    self.wikidata_triplets_df.at[index, "p_value"] = q_label

        for index, row in tqdm(
            self.wikidata_triplets_df.iterrows(),
            total=len(self.wikidata_triplets_df),
            desc="Translating Wikibase relations",
            unit="row",
        ):
            if row["p_id"] in p_value_to_label:
                p_label = p_value_to_label.get(row["p_id"])
                self.wikidata_triplets_df.at[index, "p_name"] = p_label

        return self.wikidata_triplets_df

    def get_q_p_value_to_label(self) -> dict:
        """
        Create a dictionary to translate Wikibase items.

        This function reads the Wikidata dump file line by line, parses the JSON data,
        and creates a dictionary to translate Wikibase items.

        Returns:
            dict: The dictionary to translate Wikidata items.
            dict: The dictionary to translate Wikidata relations
        """

        q_value_to_label = {}
        p_value_to_label = {}

        with bz2.open(self.wikidata_dump_path, mode="rb") as f:
            with tqdm(
                desc="Creating the dictionary to translate wikibase items", unit="line"
            ) as pbar:
                for _, line in enumerate(f):
                    try:
                        line_str = line.decode("utf-8").rstrip(",\n")
                        if line_str in ["[", "]"] or len(line_str) == 0:
                            continue

                        item = json.loads(line_str)

                        if item.get("type") == "item":
                            q_id = item["id"]
                            q_label = item.get("labels", {}).get("en", {}).get("value")
                            if q_label:
                                q_value_to_label[q_id] = q_label

                        if item.get("type") == "property":
                            p_id = item["id"]
                            p_label = item.get("labels", {}).get("en", {}).get("value")
                            if p_label:
                                p_value_to_label[p_id] = p_label

                        pbar.update(1)

                    except KeyError:
                        # Handle missing expected keys in the property data
                        continue

                    except ValueError:
                        continue

        return q_value_to_label, p_value_to_label

    def save_to_csv(self) -> None:
        """
        Save the extracted data to a CSV file.

        This function saves the extracted data to a CSV file.
        """

        self.wikidata_triplets_df.to_csv(self.output_path, index=False)
