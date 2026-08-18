"""Offline oversampling of the minority class.

The three augmented rows of the imbalance comparison differ only in which family of
transforms produces the extra minority images, so the family is a named strategy and
everything else -- how many copies, which sources, the file layout -- is shared:

    "none"          no copies, the split is returned untouched (baseline row)
    "geometric"     ShiftScaleRotate
    "photometric"   brightness/contrast/gamma and Gaussian noise
    "combined"      both

This runs once, ahead of training. The copies land on disk and are appended to the
train split as ordinary rows, so GPUDataset picks them up through the usual
image_path column and nothing in the training loop knows augmentation happened.

Copies are generated at native resolution (the 224px cache downscales afterwards, as
it does for the originals) and written as JPEG at the quality of the source set, so
that neither resolution nor container format is itself a cue for the minority label.
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image

JPEG_QUALITY = 95
CONTROL_SEED_OFFSET = 1_000_000

GEOMETRIC = [
    A.Affine(
        translate_percent=(-0.05, 0.05),
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=1.0,
    )
]


PHOTOMETRIC = [
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
    A.RandomGamma(gamma_limit=(80, 120), p=1),
    A.GaussNoise(std_range=(0.02, 0.08), per_channel=False, p=1),
]

STRATEGIES = {
    "none": [],
    "geometric": GEOMETRIC,
    "photometric": PHOTOMETRIC,
    "combined": GEOMETRIC + PHOTOMETRIC,
}


def build_pipeline(strategy: str, seed: int = None) -> A.Compose:
    """The pipeline for one strategy, or None for "none"."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {sorted(STRATEGIES)}")

    transforms = STRATEGIES[strategy]
    if not transforms:
        return None

    # seeding the Compose rather than the process ties a copy to its own index, so the
    # output does not depend on how the work happened to be spread over the pool
    return A.Compose(transforms, seed=seed)


def _augment(job):
    source, destination, strategy, seed = job
    image = np.asarray(Image.open(source).convert("L"), dtype=np.uint8)
    image = build_pipeline(strategy, seed)(image=image)["image"]
    Image.fromarray(image).save(destination, quality=JPEG_QUALITY)
    return destination


def _run(jobs, workers: int, overwrite: bool = True):
    """Generate the planned copies, regenerating any that a previous run left on disk.

    The names a plan produces depend only on the sources and the copy index, not on the
    strategy's randomness, so a rerun under a different seed lands on the files the old
    seed wrote. Regenerating by default keeps the images on disk in step with the flags
    that produced them; overwrite=False trades that for skipping a pass already done.
    """
    if not jobs:
        return

    if not overwrite and all(os.path.exists(destination) for _, destination, _, _ in jobs):
        print(f"Reusing {len(jobs)} augmented images already on disk")
        return

    for _, destination, _, _ in jobs:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_augment, jobs, chunksize=16))
    print(f"Wrote {len(jobs)} augmented images")


def _draw(n_draws: int, n_sources: int, seed: int) -> np.ndarray:
    """Indices of the sources to augment, as whole shuffled passes over the pool.

    Sampling with replacement would copy some images five times and others none;
    cycling through a reshuffled pool spreads the copies evenly (+-1 each).
    """
    passes = [
        np.random.default_rng(seed + p).permutation(n_sources)
        for p in range(math.ceil(n_draws / n_sources))
    ]
    return np.concatenate(passes)[:n_draws]


def oversample(
    df: pd.DataFrame,
    strategy: str,
    output_dir: str,
    seed: int = 41,
    workers: int = 10,
    artifact_control: bool = False,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Balance df by adding augmented copies of the minority class.

    Returns the extended split with two extra columns: augmented, and source_path for
    the file a copy came from. strategy "none" returns df unchanged.

    artifact_control additionally replaces majority images with augmented versions of
    themselves, at the same rate as the minority class ends up augmented, so that
    "looks resampled" no longer correlates with the label. It leaves the class balance
    alone -- it substitutes majority images rather than adding any.

    overwrite=False reuses the images a previous run left on disk instead of
    regenerating them, which is only safe when nothing affecting them has changed.
    """
    df = df.copy()
    df["augmented"] = False
    df["source_path"] = pd.NA

    if strategy == "none":
        return df

    counts = df["encoded_label"].value_counts()
    minority, majority = counts.idxmin(), counts.idxmax()
    n_needed = int(counts.max() - counts.min())

    pool = df[df["encoded_label"] == minority].reset_index(drop=True)
    jobs, rows = [], []

    for copy_index, source_index in enumerate(_draw(n_needed, len(pool), seed)):
        source = pool.iloc[source_index]
        destination = f"{output_dir}/{strategy}/{source['label']}/{Path(source['image_path']).stem}_aug{copy_index:05d}.jpeg"

        jobs.append((source["image_path"], destination, strategy, seed + copy_index))
        # the copy inherits the patient it came from, so a later regroup cannot put a
        # patient's original and its copies on opposite sides of a split
        rows.append({**source, "image_path": destination, "augmented": True, "source_path": source["image_path"]})

    if artifact_control:
        # after balancing, this share of the minority class is augmented; match it
        rate = n_needed / (len(pool) + n_needed)
        control = df.index[df["encoded_label"] == majority].to_numpy()
        chosen = control[_draw(round(rate * len(control)), len(control), seed + CONTROL_SEED_OFFSET)]

        for control_index, row_index in enumerate(np.unique(chosen)):
            source = df.loc[row_index]
            destination = f"{output_dir}/{strategy}/control_{source['label']}/{Path(source['image_path']).stem}_aug.jpeg"

            jobs.append((source["image_path"], destination, strategy, seed + CONTROL_SEED_OFFSET + control_index))
            df.loc[row_index, ["image_path", "augmented", "source_path"]] = [destination, True, source["image_path"]]

    _run(jobs, workers, overwrite)

    return (
        pd.concat([df, pd.DataFrame(rows)])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--csv-dir", default="data/data_csv")
    parser.add_argument("--image-dir", default="data/augmented")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--artifact-control", action="store_true")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="regenerate images already on disk (--no-overwrite reuses them)",
    )
    args = parser.parse_args()

    suffix = "_control" if args.artifact_control else ""
    out_dir = f"data/data_csv_oversampled_{args.strategy}{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    df_train = oversample(
        pd.read_csv(f"{args.csv_dir}/train.csv"),
        strategy=args.strategy,
        output_dir=args.image_dir,
        seed=args.seed,
        workers=args.workers,
        artifact_control=args.artifact_control,
        overwrite=args.overwrite,
    )
    df_train.to_csv(f"{out_dir}/train.csv", index=False)

    # val and test ride along unchanged so a run can point at one folder for all three
    for split in ("val", "test"):
        pd.read_csv(f"{args.csv_dir}/{split}.csv").to_csv(f"{out_dir}/{split}.csv", index=False)

    print(f"{out_dir}/train.csv: {len(df_train)} rows")
    print(df_train.groupby(["label", "augmented"]).size())
