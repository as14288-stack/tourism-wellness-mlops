# Data registration script: validate expected columns and print a dataset summary.
"""
Data Registration
-----------------
Reads tourism.csv from the repository's data folder, validates that every
expected column is present with a sane dtype, and prints a short summary
of the dataset so the pipeline log gives a clear record of what went in.
"""

import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Dataset not found at '{path}'. Add tourism.csv to the data folder first.")

    df = pd.read_csv(path)

    # Drop the stray index column some exports of this CSV carry, if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        sys.exit(f"Dataset is missing expected column(s): {missing_columns}")

    extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file        : {path}")
    print(f"Rows x Columns      : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns ok : {len(EXPECTED_COLUMNS) - len(missing_columns)}/{len(EXPECTED_COLUMNS)}")
    if extra_columns:
        print(f"Extra columns found : {extra_columns}")
    print("-" * 60)
    print("Missing values per column:")
    print(df[EXPECTED_COLUMNS].isnull().sum().to_string())
    print("-" * 60)
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).rename("proportion").to_string())
    print("-" * 60)
    print("Dtypes:")
    print(df[EXPECTED_COLUMNS].dtypes.to_string())
    print("=" * 60)
    print("Dataset registered successfully.")

    return df


if __name__ == "__main__":
    register_dataset()
