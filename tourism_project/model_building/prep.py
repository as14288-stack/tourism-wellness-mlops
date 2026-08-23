# Data preparation script: clean the data, split into train/test, save the splits locally.
"""
Data Preparation
----------------
Loads tourism.csv from the repository data folder, cleans it up, and
splits it into train/test sets. The splits are saved locally as CSV
files so the GitHub Actions workflow can hand them to the next job as
a workflow artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

TARGET = "ProdTaken"

# Columns that don't carry predictive information and should be dropped.
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unnecessary/identifier columns that carry no predictive signal.
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Fix an inconsistent category label found in the raw data
    # ("Fe Male" is a typo for "Female").
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Drop exact duplicate rows, if any.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate row(s).")

    return df


def split_and_save(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Columns dropped       : {COLUMNS_TO_DROP}")
    print(f"Train set             : Xtrain.csv {Xtrain.shape}, ytrain.csv {ytrain.shape}")
    print(f"Test set              : Xtest.csv {Xtest.shape}, ytest.csv {ytest.shape}")
    print(f"Train target balance  :\n{ytrain.value_counts(normalize=True).to_string()}")
    print(f"Test target balance   :\n{ytest.value_counts(normalize=True).to_string()}")
    print("=" * 60)
    print("Train/test splits saved locally: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    cleaned = load_and_clean()
    split_and_save(cleaned)
