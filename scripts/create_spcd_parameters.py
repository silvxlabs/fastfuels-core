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
ref_species_fname = "REF_SPECIES_original.csv"
output_dir = "../fastfuels_core/data"
output_fname = "REF_SPECIES.csv"

# Initialize colorama
init(autoreset=True)


# Helper function to extract the trait score from the poorly formatted table s3
# in the Purves et al data supplement.
def extract_first_value(score):
    match = re.search(r"[-+]?\d*\.\d+|\d+", score)
    return float(match.group()) if match else None


# Main function
def main():
    print(
        Fore.YELLOW
        + "\n🔥 FastFuels Core Data Processing Script 🔥\n"
        + Style.RESET_ALL
    )

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
        purves.rename(
            columns={"first_score": "PURVES_TRAIT_SCORE", "code": "SPCD"}, inplace=True
        )

        # Merge the Purves trait score with the FIA species data on the SPCD column.
        # This assigns a NoN trait score to species not found in the Purves data.
        fia_species = fia_species.merge(
            purves, how="left", left_on="SPCD", right_on="SPCD"
        )

        # Compute the mean trait score by jenkins species group
        fia_species_grouped = (
            fia_species.groupby("JENKINS_SPGRPCD")["PURVES_TRAIT_SCORE"]
            .mean()
            .reset_index()
        )
        fia_species = fia_species.merge(
            fia_species_grouped, on="JENKINS_SPGRPCD", suffixes=("", "_mean")
        )
        fia_species["PURVES_TRAIT_SCORE"] = fia_species["PURVES_TRAIT_SCORE"].fillna(
            fia_species["PURVES_TRAIT_SCORE_mean"]
        )
        fia_species.drop(columns=["PURVES_TRAIT_SCORE_mean", "CN"], inplace=True)

        # Convert the SPCD and JENKINS_SPGRPCD columns to integers
        fia_species["SPCD"] = fia_species["SPCD"].astype(int)
        fia_species["JENKINS_SPGRPCD"] = fia_species["JENKINS_SPGRPCD"].astype(int)

        # Write the dataframe to a CSV file
        output_path = Path(os.path.join(output_dir, output_fname))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fia_species.to_csv(output_path, index=True)

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
