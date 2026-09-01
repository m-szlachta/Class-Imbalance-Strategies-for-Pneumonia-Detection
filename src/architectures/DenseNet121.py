"""The DenseNet121 / ImageNet arm of the grid.

Held constant against the custom CNN in everything but the backbone: same splits, same seeds,
same training budget, same callbacks. What it does *not* share is input normalisation -- the
ImageNet weights were fitted on 3-channel input standardised with the ImageNet statistics, and
handing them the CNN's [-1, 1] tensors would penalise the pretraining arm for a reason that has
nothing to do with pretraining. See NORMALIZATIONS in src/utils/dataset.py.

The whole network is fine-tuned: nothing is frozen, the pretrained weights are the starting
point rather than a fixed feature extractor.
"""

import sys
from pathlib import Path

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.experiment import ExperimentConfig, run_experiment

# each split variant caches under its own name, so pointing DATA_DIR at an
# oversampled or undersampled split builds that split's cache rather than reusing another's
DATA_DIR = "data/data_csv"

CW = False

CONFIG = ExperimentConfig(
    model_name="DenseNet121",
    data_dir=DATA_DIR,
    cw=CW,
    # what DenseNet121_Weights.IMAGENET1K_V1.transforms() specifies; not the CNN's [-1, 1]
    normalization="imagenet",
    seeds=(0, 1, 2, 3, 4),
    deterministic=True,
)


def build_model():
    """ImageNet-pretrained DenseNet121 with a single-logit head, every parameter trainable.

    The 1000-way classifier is replaced rather than adapted, so its weights are fresh at each
    seed -- run_seed calls set_seed before this, which is what makes that draw the seed's.
    """
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    # one logit, to match BCEWithLogitsLoss; nothing applies a sigmoid before the metrics do
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    return model


if __name__ == "__main__":
    run_experiment(build_model, CONFIG)
