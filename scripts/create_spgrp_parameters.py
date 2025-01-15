import os
import json
import pandas as pd
from scipy.special import beta


def process_jenkins_spgrp_data(
    spgrp_params_file, ref_species_file, output_dir, output_filename
):
    """
    Process Jenkins species group data from parameter and reference files.
    Creates a DataFrame indexed by JENKINS_SPGRPCD containing all relevant parameters.

    Parameters:
    -----------
    spgrp_params_file : str
        Path to the JSON file containing species group parameters
    ref_species_file : str
        Path to the CSV file containing reference species data
    output_dir : str
        Directory where the output file will be saved
    output_filename : str
        Name of the output file

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by JENKINS_SPGRPCD containing all parameters
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the JSON parameters file
    with open(spgrp_params_file, "r") as f:
        spgrp_params = json.load(f)

    # Precompute BETA_NORM using Beta function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beta.html#rcac2d32f2bd2-1)
    for spgrp, params in spgrp_params.items():
        a = params["BETA_CANOPY_a"]
        b = params["BETA_CANOPY_b"]
        params["BETA_CANOPY_NORM"] = beta(a, b)

    # Convert JSON to DataFrame
    params_df = pd.DataFrame.from_dict(spgrp_params, orient="index")
    params_df.index = params_df.index.astype(float)

    # Read the reference species CSV
    ref_species_df = pd.read_csv(ref_species_file)

    # Select only Jenkins-related columns
    jenkins_columns = [col for col in ref_species_df.columns if "JENKINS_" in col]
    jenkins_df = ref_species_df[jenkins_columns].copy()

    # Group by JENKINS_SPGRPCD and take first non-null value for each column
    grouped_df = jenkins_df.groupby("JENKINS_SPGRPCD").first().copy()

    # Merge with parameters DataFrame
    final_df = grouped_df.join(params_df)

    # Ensure all expected columns are present
    expected_columns = [
        "JENKINS_TOTAL_B1",
        "JENKINS_TOTAL_B2",
        "JENKINS_STEM_WOOD_RATIO_B1",
        "JENKINS_STEM_WOOD_RATIO_B2",
        "JENKINS_STEM_BARK_RATIO_B1",
        "JENKINS_STEM_BARK_RATIO_B2",
        "JENKINS_FOLIAGE_RATIO_B1",
        "JENKINS_FOLIAGE_RATIO_B2",
        "JENKINS_ROOT_RATIO_B1",
        "JENKINS_ROOT_RATIO_B2",
        "JENKINS_SAPLING_ADJUSTMENT",
        "BETA_CANOPY_a",
        "BETA_CANOPY_b",
        "BETA_CANOPY_c",
        "BETA_CANOPY_NORM",
        "FOLIAGE_SAV",
    ]

    for col in expected_columns:
        if col not in final_df.columns:
            final_df[col] = None

    # Reorder columns
    final_df = final_df[expected_columns]

    # Set JENKINS_SPGRPCD as int
    final_df.index = final_df.index.astype(int)

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    final_df.to_csv(output_path)

    return final_df


# Example usage:
if __name__ == "__main__":
    spgrp_params_file = "data/spgrp_parameters.json"
    ref_species_file = "data/REF_SPECIES_original.csv"
    output_dir = "../fastfuels_core/data"
    output_filename = "REF_JENKINS.csv"

    result_df = process_jenkins_spgrp_data(
        spgrp_params_file, ref_species_file, output_dir, output_filename
    )
