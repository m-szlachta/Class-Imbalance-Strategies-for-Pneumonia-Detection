# Class-imbalance strategies for pneumonia detection — experiment code

Experiment code for the paper *Comparison of Class-Imbalance Strategies for Convolutional Neural
Networks for Pneumonia Detection in Pediatric Chest Radiographs*.

The study trains three CNN backbones on the Kermany pediatric chest X-ray dataset under several
class-imbalance strategies, with an identical training budget and identical seeds in every
configuration, so the strategies can be compared pairwise across seeds (e.g. with a Friedman test).

This is a research pipeline, not a library: every stage is a standalone script run from the repo
root, and an experiment is selected by editing a few constants at the top of a script.

---

## 1. Setup

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU (the whole training
split is uploaded to VRAM; CPU works but is slow).

```bash
uv sync    # creates .venv from pyproject.toml / uv.lock
```

Main dependencies: `torch`, `torchvision`, `torchxrayvision`, `albumentations`, `scikit-learn`,
`pandas`, `matplotlib`.

### Dataset

Download the Kermany *Chest X-Ray Images (Pneumonia)* dataset and unpack it so the layout is:

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/*.jpeg
├── val/{NORMAL,PNEUMONIA}/*.jpeg
└── test/{NORMAL,PNEUMONIA}/*.jpeg
```

> **Note:** `src/utils/preprocessing.py` and `src/utils/preprocessing_imbalance.py` contain
> absolute paths (`/home/michal/code/paper/...`). Edit `DATA_PATH`, `OUTPUT_ROOT` and the
> `folder_path=` arguments if the repo lives elsewhere.

---

## 2. Quick start

```bash
# 1. build the baseline and undersampled splits
uv run python src/utils/preprocessing.py

# 2. (optional) build the oversampled / reduced-minority splits
uv run python src/utils/augmentation.py geometric
uv run python src/utils/augmentation.py photometric
uv run python src/utils/augmentation.py combined
uv run python src/utils/preprocessing_imbalance.py

# 3. train + evaluate one experiment cell (5 seeds)
uv run python src/architectures/CNN.py
uv run python src/architectures/DenseNet121.py
uv run python src/architectures/TorchXRayVision.py
```

Results appear in `reports/<model>/<split name>[_cw]/` — see [section 6](#6-outputs).

---

## 3. Data preparation

Splits are **CSV manifests**, not copies of images. Every CSV has the columns
`image_path, label, patient_id, encoded_label` (augmented splits add `augmented, source_path`).
Labels: `NORMAL = 0`, `PNEUMONIA = 1`.

### 3.1 `preprocessing.py` — baseline split

```bash
uv run python src/utils/preprocessing.py
```

- Pools the upstream `train/` and `val/` folders and re-splits them 90 / 10 into train / val with
  `StratifiedGroupKFold`, **grouped by patient** (the patient ID is parsed from the filename), so
  no patient appears in more than one split. The upstream `test/` folder is used unchanged.
- Writes:
  - `data/data_csv/{train,val,test}.csv` — the baseline (natural imbalance, ~1:3 NORMAL:PNEUMONIA)
  - `data/data_csv_undersampled/{train,val,test}.csv` — PNEUMONIA randomly undersampled to the
    NORMAL count; same val and test.

### 3.2 `augmentation.py` — offline oversampling of the minority class

```bash
uv run python src/utils/augmentation.py <strategy> [options]
```

Adds augmented copies of NORMAL images to the train split until both classes are equal. Copies are
written to disk as JPEG (q95, native resolution) and appended to `train.csv` as ordinary rows;
val/test are copied through unchanged. Each copy keeps the `patient_id` of its source.

| strategy      | transforms (Albumentations)                                                         |
|---------------|-------------------------------------------------------------------------------------|
| `none`        | no copies (baseline)                                                                |
| `geometric`   | `Affine` — shift ±5%, scale 0.9–1.1, rotate ±10°                                     |
| `photometric` | `RandomBrightnessContrast` (±0.15), `RandomGamma` (80–120), `GaussNoise` (std 0.02–0.08) |
| `combined`    | geometric + photometric                                                             |

| option                         | default                                  | meaning |
|--------------------------------|------------------------------------------|---------|
| `--csv-dir`                    | `data/data_csv`                          | source split to oversample |
| `--image-dir`                  | `data/augmented`                         | where augmented images are written |
| `--out-dir`                    | `data/data_csv_oversampled_<strategy>`   | where the new CSVs go |
| `--seed`                       | `41`                                     | augmentation seed |
| `--workers`                    | `10`                                     | parallel processes |
| `--artifact-control`           | off                                      | also replace a matching share of PNEUMONIA images with augmented versions of themselves, so "looks augmented" doesn't predict the label. Output gets a `_control` suffix. |
| `--overwrite / --no-overwrite` | overwrite                                | `--no-overwrite` reuses images already on disk |

### 3.3 `preprocessing_imbalance.py` — reduced-minority splits

```bash
uv run python src/utils/preprocessing_imbalance.py
```

Separates the effect of *imbalance* from the effect of *few minority images*. Both variants keep
only 134 NORMAL images (~10%):

- `data_imbalanced/normal_minority/` — 134 NORMAL vs. all PNEUMONIA (~1:26)
- `data_imbalanced/balanced_small/` — 134 NORMAL vs. 134 PNEUMONIA

These can themselves be oversampled with `augmentation.py --csv-dir data_imbalanced/normal_minority
--out-dir ... --image-dir data_imbalanced/augmented`.

All data-side randomness (splitting, sampling, augmentation) is fixed at seed `41`, so every
experiment sees identical images and is scored on the same val/test sets.

---

## 4. Models

Each script in `src/architectures/` defines one backbone. All output a **single logit** trained with
`BCEWithLogitsLoss`; all are fully fine-tuned (nothing frozen).

| script               | model                                                         | pretraining | input normalisation | geometry |
|----------------------|---------------------------------------------------------------|-------------|---------------------|----------|
| `CNN.py`             | custom 4-block CNN (32→64→128→256 ch, GroupNorm, MaxPool) + MLP head (256→128→64→1, dropout 0.5) | none (from scratch) | `symmetric`: `[-1, 1]`, 3 channels | `squash` |
| `DenseNet121.py`     | torchvision `densenet121`, classifier replaced by `Linear(1024, 1)` | ImageNet (`IMAGENET1K_V1`) | `imagenet`: `/255`, ImageNet mean/std, 3 channels | `squash` |
| `TorchXRayVision.py` | `xrv.models.DenseNet(weights="densenet121-res224-all")`, classifier replaced by `Linear(1024, 1)`, `op_threshs=None` | chest X-rays (NIH, PadChest, CheXpert, MIMIC, Google, OpenI, RSNA) | `xrv`: `[-1024, 1024]`, 1 channel | `center_crop` |

- **squash** — resize straight to 224×224, ignoring aspect ratio.
- **center_crop** — crop the long side to a centred square, then resize (TorchXRayVision's
  convention).

Each backbone gets the preprocessing its weights were trained with; within a backbone it is the
same for every imbalance strategy.

---

## 5. Running an experiment

There is no CLI or config file for training. Open an architecture script and edit the constants at
the top:

```python
DATA_DIR = "data/data_csv"   # which split folder to train on
CW = False                   # True = class weighting in the loss
CONFIG = ExperimentConfig(
    model_name="CNN",
    data_dir=DATA_DIR,
    cw=CW,
    normalization="symmetric",
    seeds=(0, 1, 2, 3, 4),
    deterministic=True,
)
```

Then run the script, e.g. `uv run python src/architectures/CNN.py`.

### Imbalance strategy → settings

| strategy                       | `DATA_DIR`                                   | `CW`    |
|--------------------------------|----------------------------------------------|---------|
| none (baseline)                | `data/data_csv`                              | `False` |
| random undersampling           | `data/data_csv_undersampled`                 | `False` |
| oversampling — geometric       | `data/data_csv_oversampled_geometric`        | `False` |
| oversampling — photometric     | `data/data_csv_oversampled_photometric`      | `False` |
| oversampling — combined        | `data/data_csv_oversampled_combined`         | `False` |
| class weighting                | `data/data_csv`                              | `True`  |
| severe imbalance (134 NORMAL)  | `data_imbalanced/normal_minority`            | `False` |
| small balanced (134 / 134)     | `data_imbalanced/balanced_small`             | `False` |

Class weighting sets `pos_weight = n_NORMAL / n_PNEUMONIA` in `BCEWithLogitsLoss`, computed from the
train split.

### Training protocol (identical everywhere)

Defined in `ExperimentConfig` (`src/utils/experiment.py`) and `src/utils/callbacks.py`; deliberately
not tuned per configuration.

| setting          | value |
|------------------|-------|
| optimiser        | Adam, lr `1e-4`, weight decay `1e-4` |
| batch size       | 64 |
| max epochs       | 20 |
| early stopping   | patience 5 on validation loss |
| LR schedule      | ×0.5 after 2 epochs without val-loss improvement, min `1e-6` |
| checkpoint       | best validation loss; final metrics come from this checkpoint |
| decision threshold | logit `0.0` (p > 0.5) |
| seeds            | `0, 1, 2, 3, 4` |
| input size       | 224 × 224 grayscale |

Seed *k* fixes weight initialisation and batch order identically in every configuration, so
strategy A vs. B at seed *k* is a paired comparison. `deterministic=True` enables deterministic
cuDNN/cuBLAS kernels.

### Image cache

On first use, each split is decoded once into a uint8 array at `data/cache/<split dir>_<split>_224[_center_crop].npy`
and then loaded straight into GPU memory. The cache is rebuilt automatically if the CSV is newer or
the row count changes; delete `data/cache/` to force a rebuild.

---

## 6. Outputs

Each run writes to `reports/<model_name>/<DATA_DIR basename>[_cw]/`.
**This folder is deleted and recreated on every run** — copy it elsewhere if you want to keep an
old result.

```
reports/CNN/data_csv/
├── run_config.json          # everything about the run: split, normalisation, budget, sizes, device
├── seed_0/
│   ├── model.pt             # best-checkpoint state_dict
│   ├── history.csv          # per-epoch train/val/test loss and metrics
│   ├── curves.png           # loss, accuracy, AUROC, AUPRC over epochs
│   ├── metrics.json         # final val + test metrics (best checkpoint)
│   └── report.txt           # the same, human-readable, with epochs run
├── seed_1/ … seed_4/
├── seeds_test_metrics.csv   # one row per seed, test set  → input for statistical tests
├── seeds_val_metrics.csv    # one row per seed, validation set
├── summary.csv              # mean ± SD of each metric over seeds (test and val)
├── summary.txt              # readable version of the above
├── mean_curves.png          # training curves averaged over seeds (mean ± SD band)
└── summary_metrics.png      # bar chart of test metrics, mean ± SD
```

### Metrics

PNEUMONIA is the positive class (and the majority); NORMAL is the minority.

| metric              | meaning |
|---------------------|---------|
| `loss`              | BCE loss (with `pos_weight` if class weighting is on) |
| `accuracy`          | fraction correct |
| `auroc`             | area under the ROC curve |
| `auprc`             | average precision, PNEUMONIA as positive |
| `auprc_normal`      | average precision, NORMAL as positive (the minority view) |
| `sensitivity`       | recall of PNEUMONIA (TP / (TP + FN)) |
| `specificity`       | recall of NORMAL (TN / (TN + FP)) |
| `balanced_accuracy` | mean of sensitivity and specificity |
| `mcc`               | Matthews correlation coefficient |
| `f1`                | F1 score for PNEUMONIA |
| `tn, fp, fn, tp`    | confusion-matrix counts |

Threshold metrics use logit `0.0`. No threshold tuning is performed.

### Loading a trained model

```python
import torch
from src.architectures.DenseNet121 import build_model

model = build_model()
model.load_state_dict(torch.load("reports/DenseNet121/data_csv/seed_0/model.pt", weights_only=True))
model.eval()
# input: [B, 3, 224, 224] normalised as in section 4; output: [B, 1] logits (sigmoid → P(PNEUMONIA))
```

Run from the repo root so the `src.*` imports resolve.

---

## 7. Repository layout

```
src/
├── architectures/
│   ├── CNN.py                     # custom CNN + its experiment constants
│   ├── DenseNet121.py             # ImageNet DenseNet121
│   └── TorchXRayVision.py         # chest-X-ray DenseNet121
└── utils/
    ├── preprocessing.py           # patient-grouped split, undersampling
    ├── preprocessing_imbalance.py # normal_minority / balanced_small splits
    ├── augmentation.py            # offline minority oversampling
    ├── dataset.py                 # GPUDataset: cache, normalisation, geometry, batching
    ├── experiment.py              # ExperimentConfig, seed loop, training, aggregation
    ├── callbacks.py               # EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    ├── evaluation_metrics.py      # compute_metrics, PredictionCollector
    ├── evaluation.py              # MetricTracker, mean ± SD summaries and plots
    └── seeding.py                 # set_seed, determinism switches
```

Generated (not tracked): `data/`, `data_imbalanced/`, `reports/`.
