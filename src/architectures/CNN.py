"""Train the custom CNN on one imbalance strategy, five seeds deep.

One invocation covers one cell of the paper's grid: the (DATA_DIR, CW) pair set below, trained
once per seed in SEEDS. Everything a cell needs -- per-seed checkpoints, histories, curves and
metrics, plus the mean +/- SD across the seeds -- lands in reports/<run name>/, which is wiped
and rewritten on every invocation.

The seeds are the point: with the same seed pinning initialisation and batch order in every
cell, strategy A vs strategy B at seed 0 is a paired comparison, and five of those are what a
Friedman test consumes. seeds_test_metrics.csv is the file that test reads.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.evaluation import (
    MetricTracker,
    format_summary,
    plot_mean_curves,
    plot_metric_summary,
    summarize_seed_metrics,
)
from src.utils.evaluation_metrics import PredictionCollector, format_metrics
from src.utils.dataset import GPUDataset
from src.utils.seeding import configure_determinism, enable_cublas_determinism, set_seed
from src.utils.callbacks import *

BATCH_SIZE = 64
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

# each split variant caches under its own name, so pointing DATA_DIR at an
# oversampled or undersampled split builds that split's cache rather than reusing another's
DATA_DIR = "data/data_csv"

CW = False

# one run per seed; the same list in every cell is what makes the cells comparable pairwise
SEEDS = [0, 1, 2, 3, 4]

# bit-for-bit reproducibility, at some cost in throughput. Turn off if a run is too slow --
# the statistics stay valid on seeding alone, only the exact re-run goes away
DETERMINISTIC = True

REPORTS_ROOT = "reports/CNN"

EPOCHS = 20
LR = 0.0001
WEIGHT_DECAY = 1e-4

# the panels drawn in every curve plot (loss is always added in front)
CURVE_METRICS = ["accuracy", "auroc", "auprc"]

# the confusion matrix is a numpy array, so it stays in the text report and out of the CSVs
_UNSERIALISABLE = ("confusion_matrix",)


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


def train(data, model, loss_fn, optimizer, tracker, epoch):
    # the collector keeps batches on the GPU; calling .item() per batch would sync every step
    collector = PredictionCollector()
    model.train()

    for X, y in data.batches(BATCH_SIZE, shuffle=True):
        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        collector.update(pred.detach(), y, loss.detach())

    metrics = collector.compute()
    print(format_metrics(metrics, "Train"))
    tracker.record("train", epoch, metrics["loss"], metrics)

def evaluate(data, model, loss_fn):
    """Score one split without touching the tracker."""
    collector = PredictionCollector()
    model.eval()
    with torch.no_grad():
        for X, y in data.batches(BATCH_SIZE):
            pred = model(X).squeeze(1)
            collector.update(pred, y, loss_fn(pred, y))

    return collector.compute()

def validation(data, model, loss_fn, tracker, epoch):
    metrics = evaluate(data, model, loss_fn)
    print(format_metrics(metrics, "Validation"))
    tracker.record("val", epoch, metrics["loss"], metrics)

    return metrics["loss"], metrics

def test(data, model, loss_fn, tracker, epoch):
    metrics = evaluate(data, model, loss_fn)
    print(format_metrics(metrics, "Test"), "\n")
    tracker.record("test", epoch, metrics["loss"], metrics)

    return metrics


def run_name() -> str:
    """The reports folder for this cell.

    Class weighting reuses its baseline's split, so DATA_DIR alone would put the weighted and
    unweighted runs in the same folder; the suffix keeps them apart.
    """
    return Path(DATA_DIR).name + ("_cw" if CW else "")


def scalars(metrics: dict) -> dict:
    """The part of a metrics dict that survives into a CSV or JSON."""
    return {k: v for k, v in metrics.items() if k not in _UNSERIALISABLE}


def run_seed(seed, train_data, val_data, test_data, class_weight, seed_dir: Path) -> dict:
    """Train one model end to end and write everything it produced under seed_dir.

    Returns the final validation and test metrics, scored on the reloaded best checkpoint at the
    default logit threshold of 0 (p > 0.5), plus the tracker holding the epoch history.
    """
    # before the model exists: this is what pins weight init, dropout and the shuffle order.
    # Re-seeding here rather than once above the loop is what makes seed k comparable across
    # cells -- otherwise run 2 would start from wherever run 1 left the generators
    set_seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = CNN().to(DEVICE)

    if CW:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    lr = LR
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=WEIGHT_DECAY)

    model_path = seed_dir / "model.pt"
    tracker = MetricTracker()
    early_stop = EarlyStopping()
    reduce_lr = ReduceLROnPlateau()
    model_checkpoint = ModelCheckpoint(str(model_path))

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data, model, loss_fn, optimizer, tracker, t + 1)
        vloss, _ = validation(val_data, model, loss_fn, tracker, t + 1)

        model_checkpoint.save_model(model, vloss)
        test(test_data, model, loss_fn, tracker, t + 1)

        if early_stop.on_epoch_end(vloss):
            break

        lr = reduce_lr.reduce(vloss, lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    model.load_state_dict(torch.load(model_path, weights_only=True))

    # the reported numbers come from the best checkpoint, not the last epoch. They stay out of
    # the tracker: recording them would put a second point on the curve at an epoch that
    # already has one
    final_val = evaluate(val_data, model, loss_fn)
    final_test = evaluate(test_data, model, loss_fn)

    print(f"Seed {seed} best checkpoint:")
    print(format_metrics(final_val, "Validation"))
    print(format_metrics(final_test, "Test"), "\n")

    tracker.to_csv(str(seed_dir / "history.csv"))
    tracker.plot(str(seed_dir / "curves.png"), metrics=CURVE_METRICS)

    (seed_dir / "metrics.json").write_text(
        json.dumps({"seed": seed, "val": scalars(final_val), "test": scalars(final_test)}, indent=2)
    )
    (seed_dir / "report.txt").write_text(
        "\n".join(
            [
                f"seed {seed} | {DATA_DIR} | class weighting: {CW}",
                f"epochs run: {len(tracker.history['train']['epoch'])} of {EPOCHS}",
                "",
                format_metrics(final_val, "Validation (best checkpoint)"),
                "",
                format_metrics(final_test, "Test (best checkpoint)"),
                "",
            ]
        )
    )

    return {"val": final_val, "test": final_test, "history": tracker.history}


def aggregate(results, out_dir: Path) -> None:
    """Turn the per-seed results into the numbers and plots the paper reports."""
    summaries = []
    for split in ("test", "val"):
        rows = [{"seed": seed, **scalars(result[split])} for seed, result in results]
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"seeds_{split}_metrics.csv", index=False)

        summary = summarize_seed_metrics(rows, split=split)
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv(out_dir / "summary.csv", index=False)

    test_summary = summaries[0]
    plot_metric_summary(
        test_summary,
        str(out_dir / "summary_metrics.png"),
        title=f"{out_dir.name} -- test metrics (mean +/- SD, n={len(results)})",
    )
    plot_mean_curves(
        [result["history"] for _, result in results],
        str(out_dir / "mean_curves.png"),
        metrics=CURVE_METRICS,
    )

    epochs_run = ", ".join(
        f"seed {seed}: {len(result['history']['train']['epoch'])}" for seed, result in results
    )
    text = "\n".join(
        [
            f"{out_dir.name} | {DATA_DIR} | class weighting: {CW}",
            f"seeds: {[seed for seed, _ in results]}",
            f"epochs run: {epochs_run}",
            "",
            format_summary(summaries[0], "Test (best checkpoint, threshold 0.0)"),
            "",
            format_summary(summaries[1], "Validation (best checkpoint, threshold 0.0)"),
            "",
        ]
    )
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)


def main():
    enable_cublas_determinism()
    configure_determinism(DETERMINISTIC)

    # the decode is RNG-free, so the splits are loaded once and shared by every seed
    train_data = GPUDataset(f"{DATA_DIR}/train.csv", DEVICE)
    val_data = GPUDataset(f"{DATA_DIR}/val.csv", DEVICE)
    test_data = GPUDataset(f"{DATA_DIR}/test.csv", DEVICE)

    n_pos = train_data.labels.sum()
    class_weight = (len(train_data) - n_pos) / n_pos

    out_dir = Path(REPORTS_ROOT) / run_name()
    # rewritten wholesale: a shorter SEEDS list must not leave the previous run's seed folders
    # sitting next to the new ones
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": DATA_DIR,
                "class_weighting": CW,
                "class_weight": float(class_weight),
                "seeds": SEEDS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "deterministic": DETERMINISTIC,
                "device": DEVICE,
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
            },
            indent=2,
        )
    )

    results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ({DATA_DIR}, CW={CW}) ===")
        result = run_seed(seed, train_data, val_data, test_data, class_weight, out_dir / f"seed_{seed}")
        results.append((seed, result))

    aggregate(results, out_dir)
    print(f"\nWrote {len(results)} seed(s) to {out_dir}")


if __name__ == "__main__":
    main()
