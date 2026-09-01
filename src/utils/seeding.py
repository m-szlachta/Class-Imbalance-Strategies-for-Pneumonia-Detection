"""Reproducibility controls shared by every architecture script.

The paper reports mean +/- SD over 5 seeds per cell and pairs the cells seed-by-seed, so the
seed has to be an explicit input rather than whatever PyTorch initialises itself with. Call
set_seed immediately before each run -- seeding once and then looping leaves run 2 starting from
wherever run 1 left the generators, which breaks the pairing.

There is no DataLoader in this pipeline: GPUDataset.batches shuffles with torch.randperm on the
device, so torch.manual_seed covers the shuffle order and there are no workers to seed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Pin every generator a run draws from: weight init, dropout and the batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool) -> None:
    """Trade throughput for a seed that reproduces bit-for-bit.

    warn_only=True so an op without a deterministic kernel warns instead of aborting a job that
    is five runs deep. cudnn.benchmark has to go the other way from the default: autotuning picks
    an algorithm from timings, which is not stable across runs.
    """
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def enable_cublas_determinism() -> None:
    """Set the cuBLAS workspace config deterministic matmuls need.

    cuBLAS reads this once, when its handle is created, so it has to already be in the
    environment before the first CUDA operation -- call this at the top of a script, above the
    point where the datasets are uploaded to the device. Importing torch alone does not create
    the context, so it is enough to be early rather than to precede the import.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
