import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "/home/michal/code/paper/data/chest_xray"

def create_dataframes(data_path:str):
    root = Path(data_path)

    train_rows, test_rows = [], []

    for img in root.rglob("*.jpeg"):
        label = img.parent.name          # NORMAL / PNEUMONIA
        split = img.parent.parent.name   # train / val / test

        row = {"image_path": str(img), "label": label}
        if split in ("train", "val"):
            train_rows.append(row)
        elif split == "test":
            test_rows.append(row)

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    df_train["encoded_label"] = pd.factorize(df_train["label"])[0] #encoding normal = 0, pneumonia = 1
    df_test["encoded_label"] = pd.factorize(df_test["label"])[0] #encoding normal = 0, pneumonia = 1
    return df_train, df_test

def data_split(df, ratio = 0.1, random_state = 41):
    df_train, df_val = train_test_split(df, test_size=ratio, random_state=random_state, stratify=df["encoded_label"])

    return df_train, df_val

def save_to_csv(dataframes: dict, folder_path: str):
    os.makedirs(folder_path, exist_ok=True)
    for name, dataframe in dataframes.items():
        dataframe.to_csv(f"{folder_path}/{name}.csv", index=False)


df_train, df_test = create_dataframes(DATA_PATH)
df_train, df_val = data_split(df_train)

save_to_csv(
    dataframes={"train": df_train, "val": df_val, "test": df_test},
    folder_path="/home/michal/code/paper/data/data_csv",
)
