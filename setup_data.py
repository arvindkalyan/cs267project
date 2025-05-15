#!/usr/bin/env python3
"""
Utility script to check and organize Facebook ego network data files
"""

import os
import sys
import glob
import tarfile
import shutil


def extract_and_organize_data(
    data_dir="./data/facebook", tar_file="./data/facebook.tar.gz"
):
    """
    Check for Facebook data files, extract if necessary, and organize

    Args:
        data_dir: Directory containing the Facebook data
    """
    print("Checking for Facebook ego network data files...")

    if os.path.exists(tar_file):
        print(f"Found facebook.tar.gz file at {tar_file}")

        # Check if already extracted
        edge_files = glob.glob(os.path.join(data_dir, "*.edges"))

        if edge_files:
            print("Data files already extracted.")
        else:
            print("Extracting data files...")
            # Extract the tar file
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path="./data")
            print("Extraction complete.")
    else:
        print(f"Could not find facebook.tar.gz at {tar_file}")
        print(
            "Please download the data from: https://snap.stanford.edu/data/ego-Facebook.html and place it in the data directory."
        )
        return False

    # Check for expected file types
    ego_ids = []

    for file_path in glob.glob(os.path.join(data_dir, "*.edges")):
        ego_id = os.path.basename(file_path).split(".")[0]
        ego_ids.append(ego_id)

    if not ego_ids:
        print("No .edges files found in the data directory.")
        return False

    print(f"Found {len(ego_ids)} ego networks:")
    for i, ego_id in enumerate(ego_ids):
        print(f"  - Ego ID: {ego_id}")

    # Check for required file types
    missing_files = []

    for ego_id in ego_ids:
        # Check for edges file
        if not os.path.exists(os.path.join(data_dir, f"{ego_id}.edges")):
            missing_files.append(f"{ego_id}.edges")

        # Check for features file
        if not os.path.exists(os.path.join(data_dir, f"{ego_id}.feat")):
            missing_files.append(f"{ego_id}.feat")

    if missing_files:
        print("Warning: Some expected files are missing:")
        for file in missing_files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print("All expected files are present.")

    return True


def main():
    # Get data directory from command line arguments or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/facebook"
    tar_file = "./data/facebook.tar.gz"

    # Check and organize data
    success = extract_and_organize_data(data_dir, tar_file)

    if not success:
        print("\nData setup failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
