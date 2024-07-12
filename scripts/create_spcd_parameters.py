"""
This script creates the spcd_parameters.json file shipped with
fastfuels-core in the data directory of the package

NOTE: cd into the scripts directory before running this script
"""

# Core imports
import re
import json
import sys
from pathlib import Path
import os

# External imports
import pandas as pd
from colorama import Fore, Style, init

# Configurations
data_dir = "./data"
purves_fname = "purves_s3.csv"
ref_species_fname = "REF_SPECIES.csv"
output_dir = "../fastfuels_core/data"
output_fname = "spcd_parameters_test.json"

# Initialize colorama
init(autoreset=True)

# Helper function to extract the trait score from the poorly formatted table s3
# in the Purves et al data supplement.
def extract_first_value(score):
    match = re.search(r"[-+]?\d*\.\d+|\d+", score)
    return float(match.group()) if match else None


# Main function
def main():
    print(Fore.YELLOW + "\n🔥 FastFuels Core Data Processing Script 🔥\n" + Style.RESET_ALL)

    try:
        # Read source data files
        purves = pd.read_csv(os.path.join(data_dir, purves_fname))
        fia_species = pd.read_csv(os.path.join(data_dir, ref_species_fname))
    except FileNotFoundError as e:
        print(
            Fore.RED
            + f"Error: {e}. Please ensure the data files are in the correct directory."
            + Style.RESET_ALL
        )
        sys.exit(1)

    try:
        # Extract the trait score
        purves["first_score"] = purves["score"].apply(extract_first_value)
        purves.drop(columns=["score"], inplace=True)
        purves.rename(columns={"first_score": "score"}, inplace=True)

        # Merge data
        fia_species = fia_species.merge(purves, left_on="SPCD", right_on="code")

        # Compute the mean trait score by jenkins species group
        fia_species_grouped = (
            fia_species.groupby("JENKINS_SPGRPCD")["score"].mean().reset_index()
        )
        fia_species = fia_species.merge(
            fia_species_grouped, on="JENKINS_SPGRPCD", suffixes=("", "_mean")
        )
        fia_species["score"] = fia_species["score"].fillna(fia_species["score_mean"])
        fia_species.drop(columns=["score_mean"], inplace=True)

        # Set SPCD as the index
        fia_species.set_index("SPCD", inplace=True)

        # Transform CLASS column: 0 for hardwoods, 1 for softwoods
        fia_species["CLASS"] = fia_species["SFTWD_HRDWD"].apply(
            lambda x: 1 if x == "S" else 0
        )

        # Convert SPGRP and SPCD to integers
        fia_species["JENKINS_SPGRPCD"] = fia_species["JENKINS_SPGRPCD"].astype(int)
        fia_species.index = fia_species.index.astype(int)

        # Keep only the necessary columns and rename them
        fia_species = fia_species.loc[
            :, ["CLASS", "JENKINS_SPGRPCD", "JENKINS_SAPLING_ADJUSTMENT", "score"]
        ]
        fia_species.rename(
            columns={
                "JENKINS_SPGRPCD": "SPGRP",
                "JENKINS_SAPLING_ADJUSTMENT": "SAPADJ",
                "score": "PURVES_TRAIT_SCORE",
            },
            inplace=True,
        )

        # Convert SPGRP to integer after renaming
        fia_species["SPGRP"] = fia_species["SPGRP"].astype(int)

        # Convert the DataFrame to a dictionary
        fia_species_dict = fia_species.to_dict(orient="index")

        # Write the dictionary to a JSON file
        output_path = Path(os.path.join(output_dir, output_fname))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as json_file:
            json.dump(fia_species_dict, json_file, indent=4)

        print(
            Fore.GREEN
            + "✅ JSON file created successfully at "
            + str(output_path)
            + Style.RESET_ALL
        )

    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        sys.exit(1)


if __name__ == "__main__":
    main()
