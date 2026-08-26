"""Two training sets that isolate imbalance from minority-set size.

The undersampled split in preprocessing.py equalises the classes at the size the
majority allows (1214 / 1214). These two go the other way, both holding NORMAL at
134 images -- ~10% of the 1341 NORMAL radiographs in the training folder:

    "normal_minority"   134 NORMAL against the untouched ~3494 PNEUMONIA (~1:26)
    "balanced_small"    134 NORMAL against 134 PNEUMONIA

The pair differs only in how many majority images sit beside the same 134, so a gap
between them is the imbalance itself rather than the scarcity of minority images.

Only the train split is cut. val and test are written out untouched and identical
across both variants, so every cell of the comparison is scored on the same images.
"""

import pandas as pd

from preprocessing import (
    DATA_PATH,
    LABEL_MAP,
    create_dataframes,
    data_split,
    save_to_csv,
)

OUTPUT_ROOT = "/home/michal/code/paper/data_imbalanced"
# ~10% of the 1341 NORMAL images in data/chest_xray/train
N_MINORITY = 134


def cap_classes(df_train, caps: dict, random_state=41):
    """Keep at most caps[label] images of each named class; unnamed classes pass through whole.

    The general form of undersample(): capping every class at the smallest count
    reproduces it, while capping one class leaves the other at full size.
    """
    kept = [
        group.sample(n=min(caps[label], len(group)), random_state=random_state)
        if label in caps
        else group
        for label, group in df_train.groupby("encoded_label")
    ]

    return (
        pd.concat(kept)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # split once, so both variants inherit the same val and test images
    df_train, df_test = create_dataframes(DATA_PATH)
    df_train, df_val = data_split(df_train)

    variants = {
        "normal_minority": {LABEL_MAP["NORMAL"]: N_MINORITY},
        "balanced_small": {LABEL_MAP["NORMAL"]: N_MINORITY, LABEL_MAP["PNEUMONIA"]: N_MINORITY},
    }

    for name, caps in variants.items():
        df_train_capped = cap_classes(df_train, caps)

        save_to_csv(
            dataframes={"train": df_train_capped, "val": df_val, "test": df_test},
            folder_path=f"{OUTPUT_ROOT}/{name}",
        )

        counts = df_train_capped["label"].value_counts().to_dict()
        print(f"{name}: {len(df_train_capped)} train images {counts}")
