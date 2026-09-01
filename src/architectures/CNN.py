"""The custom CNN arm of the grid.

One invocation trains the (DATA_DIR, CW) cell set below once per seed and writes
reports/CNN/<run name>/. The seed loop, the callbacks and the aggregation all live in
src.utils.experiment -- this file is the model and the constants that select the cell.
"""

import sys
from pathlib import Path

from torch import nn

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.experiment import ExperimentConfig, run_experiment

# each split variant caches under its own name, so pointing DATA_DIR at an
# oversampled or undersampled split builds that split's cache rather than reusing another's
DATA_DIR = "data/data_csv"

CW = False

CONFIG = ExperimentConfig(
    model_name="CNN",
    data_dir=DATA_DIR,
    cw=CW,
    # trained from scratch, so there is no pretraining convention to honour: /127.5 - 1
    normalization="symmetric",
    seeds=(0, 1, 2, 3, 4),
    deterministic=True,
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(2, 32),      # 32 ch, 2 groups (16 ch each)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),      # 64 ch, 4 groups
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),     # 128 ch, 8 groups
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),    # 256 ch, 16 groups
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)

        return x


def build_model():
    return CNN()


if __name__ == "__main__":
    run_experiment(build_model, CONFIG)
