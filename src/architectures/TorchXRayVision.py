"""The DenseNet121 / TorchXRayVision arm of the grid.

The architecture is held constant against DenseNet121.py -- same 121-layer densenet, same splits,
same seeds, same training budget, same callbacks -- so the cell isolates the *pretraining source*:
natural images (ImageNet) against chest radiographs (nih-pc-chex-mimic_ch-google-openi-rsna, via
https://github.com/mlmed/torchxrayvision).

Its input convention differs on all three axes, and all three are honoured rather than approximated:
range ([-1024, 1024], not ImageNet's standardised [0, 1]), channels (1, not 3) and geometry
(XRayCenterCrop then resize, not a straight squash to 224x224). See NORMALIZATIONS and GEOMETRIES in
src/utils/dataset.py; center_crop keeps its own cache, so the other arms' pixels are untouched.

This file is deliberately not called torchxrayvision.py: run as a script, sys.path[0] is
src/architectures, so a lowercase name would shadow the installed package and import itself.

The whole network is fine-tuned: nothing is frozen, the pretrained weights are the starting
point rather than a fixed feature extractor.
"""

import sys
from pathlib import Path

import torchxrayvision as xrv
from torch import nn

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.experiment import ExperimentConfig, run_experiment

DATA_DIR = "data/data_csv"

CW = False

CONFIG = ExperimentConfig(
    model_name="TorchXRayVision",
    data_dir=DATA_DIR,
    cw=CW,
    # what xrv.utils.normalize produces, on the single channel xrv.models.DenseNet expects
    normalization="xrv",
    # what xrv.datasets.XRayCenterCrop does before the resize; these weights saw nothing else
    geometry="center_crop",
    seeds=(0, 1, 2, 3, 4),
    deterministic=True,
)


def build_model():
    """Chest-radiograph-pretrained DenseNet121 with a single-logit head, every parameter trainable.

    The 18-way classifier is replaced rather than adapted, so its weights are fresh at each
    seed -- run_seed calls set_seed before this, which is what makes that draw the seed's.
    """
    # the head cannot be sized here: DenseNet.__init__ refuses num_classes alongside weights,
    # because the checkpoint defines its own 18 outputs
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    # loading the weights also loads 18 operating-point thresholds, and forward() applies a
    # sigmoid and op_norm whenever they are set. That would fight BCEWithLogitsLoss -- and index
    # 18 thresholds into a 1-logit output. None restores a plain logit.
    model.op_threshs = None

    # one logit, to match BCEWithLogitsLoss; nothing applies a sigmoid before the metrics do
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    return model


if __name__ == "__main__":
    run_experiment(build_model, CONFIG)
