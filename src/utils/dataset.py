"""Decode the image set once, then keep it on the GPU.

The images are large (~1300x760) but the model wants 224x224, and the whole set is
small enough to fit in VRAM at that size. Decoding on every epoch left the GPU idle
~85% of the time, so we decode once into a uint8 cache and index it on-device.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image

IMAGE_SIZE = 224
# the x-rays are single-channel (a few are stored as RGB with identical channels),
# so we cache one channel and expand to three at batch time
CACHE_CHANNELS = 1


def _load(args):
    path, size = args
    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def build_cache(csv_path: str, cache_path: str, size: int = IMAGE_SIZE, workers: int = 10) -> str:
    """Decode every image named in csv_path into a uint8 [N, size, size] .npy cache."""
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(csv_path):
        return cache_path

    paths = pd.read_csv(csv_path)["image_path"].tolist()
    with ProcessPoolExecutor(workers) as pool:
        images = np.stack(list(pool.map(_load, ((p, size) for p in paths), chunksize=32)))

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, images)
    print(f"Cached {len(images)} images to {cache_path} ({images.nbytes / 1e6:.0f} MB)")
    return cache_path


class GPUDataset:
    """The whole split, resident in VRAM. Batches are sliced on-device, so there is no
    per-epoch host work and no DataLoader."""

    def __init__(self, csv_path: str, cache_path: str, device, size: int = IMAGE_SIZE):
        build_cache(csv_path, cache_path, size)
        labels = pd.read_csv(csv_path)["encoded_label"].values
        self.images = torch.from_numpy(np.load(cache_path)).to(device)
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
