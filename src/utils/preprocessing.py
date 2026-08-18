import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import os
import re

DATA_PATH = "/home/michal/code/paper/data/chest_xray"
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

# person45_bacteria_1996.jpeg -> patient 45 of the bacteria series.
# bacteria and virus are separately numbered, so the type is part of the identity.
PNEUMONIA_PATTERN = re.compile(r"^person(\d+)_(bacteria|virus)_")
# IM-0117-0001.jpeg and NORMAL2-IM-0117-0001.jpeg -> IM number 0117, prefix "" and "2".
NORMAL_PATTERN = re.compile(r"^(?:NORMAL(\d+)-)?IM-(\d+)-")


def patient_id(image_path: str, label: str, merge_normal_series: bool = True) -> str:

    filename = os.path.basename(image_path)

    if label == "PNEUMONIA":
        match = PNEUMONIA_PATTERN.match(filename)
        if match:
            number, infection = match.groups()
            return f"PNEUMONIA/{infection}/{number}"
    else:
        match = NORMAL_PATTERN.match(filename)
        if match:
            series, number = match.groups()
            if merge_normal_series:
                return f"NORMAL/{number}"
            return f"NORMAL/{series or ''}/{number}"

    raise ValueError(f"cannot read a patient id from {label} file {filename!r}")


def create_dataframes(data_path:str):
    root = Path(data_path)

    train_rows, test_rows = [], []

    for img in root.rglob("*.jpeg"):
        label = img.parent.name          # NORMAL / PNEUMONIA
        split = img.parent.parent.name   # train / val / test

        row = {"image_path": str(img), "label": label, "patient_id": patient_id(str(img), label)}
        if split in ("train", "val"):
            train_rows.append(row)
        elif split == "test":
            test_rows.append(row)

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    df_train["encoded_label"] = df_train["label"].map(LABEL_MAP)
    df_test["encoded_label"] = df_test["label"].map(LABEL_MAP)

    return df_train, df_test

def data_split(df, ratio = 0.1, random_state = 41):
    """Split by patient, not by image, keeping the class balance of the pool."""
    n_splits = round(1 / ratio)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, df["encoded_label"], groups=df["patient_id"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val

def undersample(df_train, random_state = 41):
    """Drop random majority-class images so both classes have the same count."""
    counts = df_train["encoded_label"].value_counts()
    n_keep = counts.min()

    kept = [
        group.sample(n=n_keep, random_state=random_state)
        for _, group in df_train.groupby("encoded_label")
    ]

    return (
        pd.concat(kept)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

def save_to_csv(dataframes: dict, folder_path: str):
    os.makedirs(folder_path, exist_ok=True)
    for name, dataframe in dataframes.items():
        dataframe.to_csv(f"{folder_path}/{name}.csv", index=False)


df_train, df_test = create_dataframes(DATA_PATH)
df_train, df_val = data_split(df_train)
df_train = undersample(df_train)

splits = {"train": df_train, "val": df_val, "test": df_test}

save_to_csv(
    dataframes=splits,
    folder_path="/home/michal/code/paper/data/data_csv_undersampled",
)
