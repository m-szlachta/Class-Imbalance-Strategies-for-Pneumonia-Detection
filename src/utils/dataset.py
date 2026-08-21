"""Decode the image set once, then keep it on the GPU.

The images are large (~1300x760) but the model wants 224x224, and the whole set is
small enough to fit in VRAM at that size. Decoding on every epoch left the GPU idle
~85% of the time, so we decode once into a uint8 cache and index it on-device.

A cache belongs to exactly one CSV and is named after it, so the baseline, undersampled
and oversampled splits each keep their own file. Pointing a run at a different split
therefore builds (or reuses) that split's cache instead of silently inheriting whichever
images happened to be cached last.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

IMAGE_SIZE = 224
CACHE_DIR = "data/cache"
# the x-rays are single-channel (a few are stored as RGB with identical channels),
# so we cache one channel and expand to three at batch time
CACHE_CHANNELS = 1


def _load(args):
    path, size = args
    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def cache_path_for(csv_path: str, size: int = IMAGE_SIZE) -> str:
    """The cache file belonging to csv_path.

    The split variants all name their file train.csv, so it is the parent directory
    (data_csv, data_csv_oversampled_geometric, ...) that tells them apart; both go in
    the name, along with the size, which changes the pixels a cache holds.
    """
    csv = Path(csv_path)
    return os.path.join(CACHE_DIR, f"{csv.parent.name}_{csv.stem}_{size}.npy")


def _is_current(cache_path: str, csv_path: str, rows: int, size: int) -> bool:
    """Whether the cache on disk can stand in for a fresh decode of csv_path."""
    if not os.path.exists(cache_path):
        return False

    # the CSV is rewritten whenever its rows change, and augmentation rewrites it
    # alongside the images it regenerates, so an older cache cannot be trusted
    if os.path.getmtime(cache_path) < os.path.getmtime(csv_path):
        return False

    # the backstop: a cache whose shape disagrees with the CSV is unusable whatever
    # the timestamps say. mmap reads the header only, not the 236 MB behind it
    return np.load(cache_path, mmap_mode="r").shape == (rows, size, size)


def build_cache(csv_path: str, cache_path: str = None, size: int = IMAGE_SIZE, workers: int = 10) -> str:
    """Decode every image named in csv_path into a uint8 [N, size, size] .npy cache."""
    cache_path = cache_path or cache_path_for(csv_path, size)
    paths = pd.read_csv(csv_path)["image_path"].tolist()

    if _is_current(cache_path, csv_path, len(paths), size):
        return cache_path

    with ProcessPoolExecutor(workers) as pool:
        images = np.stack(list(pool.map(_load, ((p, size) for p in paths), chunksize=32)))

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, images)
    print(f"Cached {len(images)} images to {cache_path} ({images.nbytes / 1e6:.0f} MB)")
    return cache_path


class GPUDataset:
    """The whole split, resident in VRAM. Batches are sliced on-device, so there is no
    per-epoch host work and no DataLoader."""

    def __init__(self, csv_path: str, device, cache_path: str = None, size: int = IMAGE_SIZE):
        cache_path = build_cache(csv_path, cache_path, size)
        labels = pd.read_csv(csv_path)["encoded_label"].values
        images = np.load(cache_path)

        # build_cache already guarantees this; failing here rather than on the first
        # batch turns a CUDA device-side assert into a sentence naming the two files
        if len(images) != len(labels):
            raise ValueError(
                f"{cache_path} holds {len(images)} images but {csv_path} has {len(labels)} rows; "
                f"delete the cache and rerun"
            )

        self.images = torch.from_numpy(images).to(device)
        self.labels = torch.tensor(labels, device=device).float()

    def __len__(self):
        return len(self.labels)

    def num_batches(self, batch_size: int) -> int:
        return (len(self) + batch_size - 1) // batch_size

    def batches(self, batch_size: int, shuffle: bool = False):
        device = self.images.device
        order = torch.randperm(len(self), device=device) if shuffle else torch.arange(len(self), device=device)

        for start in range(0, len(self), batch_size):
            index = order[start : start + batch_size]
            # reproduces ToTensor() (/255) followed by Normalize(0.5, 0.5)
            images = self.images[index].float().div_(127.5).sub_(1.0)
            yield images.unsqueeze(1).expand(-1, 3, -1, -1), self.labels[index]
