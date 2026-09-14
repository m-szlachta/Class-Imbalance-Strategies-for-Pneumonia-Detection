"""The training harness every architecture shares.

One call to run_experiment covers one cell of the paper's grid: a (backbone, split, class
weighting) combination, trained once per seed. The architecture scripts under src/architectures
supply a model factory and their constants; everything else -- the seed loop, the callbacks, the
per-seed artefacts and the mean +/- SD aggregate -- lives here, so a fix lands once rather than
once per backbone.

The seeds are the point: with the same seed pinning initialisation and batch order in every cell,
strategy A vs strategy B at seed 0 is a paired comparison, and five of those are what a Friedman
test consumes. seeds_test_metrics.csv is the file that test reads.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.utils.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.utils.dataset import GPUDataset
from src.utils.evaluation import (
    MetricTracker,
    format_summary,
    plot_mean_curves,
    plot_metric_summary,
    summarize_seed_metrics,
)
from src.utils.evaluation_metrics import PredictionCollector, format_metrics
from src.utils.seeding import configure_determinism, enable_cublas_determinism, set_seed

# the confusion matrix is a numpy array, so it stays in the text report and out of the CSVs
_UNSERIALISABLE = ("confusion_matrix",)


@dataclass
class ExperimentConfig:
    """One cell of the grid. The defaults are the protocol, not suggestions.

    notes/what_to_use.txt fixes an identical training budget across every cell and forbids
    configuration-specific tuning, so epochs, optimiser, LR and schedule are the same everywhere;
    what a cell varies is data_dir, cw and the backbone.
    """

    model_name: str
    data_dir: str
    cw: bool
    normalization: str = "symmetric"
    geometry: str = "squash"
    seeds: tuple = (0, 1, 2, 3, 4)
    deterministic: bool = True
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    reports_root: str = "reports"
    curve_metrics: tuple = ("accuracy", "auroc", "auprc")

    def run_name(self) -> str:
        """The cell's folder. Class weighting reuses its baseline's split, so DATA_DIR alone
        would put the weighted and unweighted runs in the same place."""
        return Path(self.data_dir).name + ("_cw" if self.cw else "")

    def out_dir(self) -> Path:
        return Path(self.reports_root) / self.model_name / self.run_name()


def scalars(metrics: dict) -> dict:
    """The part of a metrics dict that survives into a CSV or JSON."""
    return {k: v for k, v in metrics.items() if k not in _UNSERIALISABLE}


def train_epoch(data, model, loss_fn, optimizer, tracker, epoch, batch_size):
    # the collector keeps batches on the GPU; calling .item() per batch would sync every step
    collector = PredictionCollector()
    model.train()

    for X, y in data.batches(batch_size, shuffle=True):
        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        collector.update(pred.detach(), y, loss.detach())

    metrics = collector.compute()
    print(format_metrics(metrics, "Train"))
    tracker.record("train", epoch, metrics["loss"], metrics)


def evaluate(data, model, loss_fn, batch_size):
    """Score one split without touching the tracker."""
    collector = PredictionCollector()
    model.eval()
    with torch.no_grad():
        for X, y in data.batches(batch_size):
            pred = model(X).squeeze(1)
            collector.update(pred, y, loss_fn(pred, y))

    return collector.compute()


def run_seed(build_model, seed, splits, class_weight, config: ExperimentConfig, seed_dir: Path) -> dict:
    """Train one model end to end and write everything it produced under seed_dir.

    Returns the final validation and test metrics, scored on the reloaded best checkpoint at the
    default logit threshold of 0 (p > 0.5), plus the epoch history behind the curves.
    """
    train_data, val_data, test_data = splits

    # before the model exists: this is what pins weight init (including a freshly attached
    # classifier head), dropout and the shuffle order. Re-seeding here rather than once above
    # the loop is what makes seed k comparable across cells -- otherwise run 2 would start from
    # wherever run 1 left the generators
    set_seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    device = train_data.images.device
    model = build_model().to(device)

    if config.cw:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=config.weight_decay)

    model_path = seed_dir / "model.pt"
    tracker = MetricTracker()
    early_stop = EarlyStopping()
    reduce_lr = ReduceLROnPlateau()
    model_checkpoint = ModelCheckpoint(str(model_path))

    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_data, model, loss_fn, optimizer, tracker, t + 1, config.batch_size)

        vmetrics = evaluate(val_data, model, loss_fn, config.batch_size)
        print(format_metrics(vmetrics, "Validation"))
        tracker.record("val", t + 1, vmetrics["loss"], vmetrics)
        vloss = vmetrics["loss"]

        model_checkpoint.save_model(model, vloss)

        tmetrics = evaluate(test_data, model, loss_fn, config.batch_size)
        print(format_metrics(tmetrics, "Test"), "\n")
        tracker.record("test", t + 1, tmetrics["loss"], tmetrics)

        if early_stop.on_epoch_end(vloss):
            break

        lr = reduce_lr.reduce(vloss, lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    model.load_state_dict(torch.load(model_path, weights_only=True))

    # the reported numbers come from the best checkpoint, not the last epoch. They stay out of
    # the tracker: recording them would put a second point on the curve at an epoch that
    # already has one
    final_val = evaluate(val_data, model, loss_fn, config.batch_size)
    final_test = evaluate(test_data, model, loss_fn, config.batch_size)

    print(f"Seed {seed} best checkpoint:")
    print(format_metrics(final_val, "Validation"))
    print(format_metrics(final_test, "Test"), "\n")

    tracker.to_csv(str(seed_dir / "history.csv"))
    tracker.plot(str(seed_dir / "curves.png"), metrics=list(config.curve_metrics))

    (seed_dir / "metrics.json").write_text(
        json.dumps({"seed": seed, "val": scalars(final_val), "test": scalars(final_test)}, indent=2)
    )
    (seed_dir / "report.txt").write_text(
        "\n".join(
            [
                f"seed {seed} | {config.model_name} | {config.data_dir} | class weighting: {config.cw}",
                f"epochs run: {len(tracker.history['train']['epoch'])} of {config.epochs}",
                "",
                format_metrics(final_val, "Validation (best checkpoint)"),
                "",
                format_metrics(final_test, "Test (best checkpoint)"),
                "",
            ]
        )
    )

    return {"val": final_val, "test": final_test, "history": tracker.history}


def aggregate(results, out_dir: Path, config: ExperimentConfig) -> None:
    """Turn the per-seed results into the numbers and plots the paper reports."""
    summaries = []
    for split in ("test", "val"):
        rows = [{"seed": seed, **scalars(result[split])} for seed, result in results]
        pd.DataFrame(rows).to_csv(out_dir / f"seeds_{split}_metrics.csv", index=False)
        summaries.append(summarize_seed_metrics(rows, split=split))

    pd.concat(summaries, ignore_index=True).to_csv(out_dir / "summary.csv", index=False)

    plot_metric_summary(
        summaries[0],
        str(out_dir / "summary_metrics.png"),
        title=f"{config.model_name} / {config.run_name()} -- test metrics (mean +/- SD, n={len(results)})",
    )
    plot_mean_curves(
        [result["history"] for _, result in results],
        str(out_dir / "mean_curves.png"),
        metrics=list(config.curve_metrics),
    )

    epochs_run = ", ".join(
        f"seed {seed}: {len(result['history']['train']['epoch'])}" for seed, result in results
    )
    text = "\n".join(
        [
            f"{config.model_name} | {config.data_dir} | class weighting: {config.cw}",
            f"normalization: {config.normalization}",
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


def run_experiment(build_model, config: ExperimentConfig) -> Path:
    """Train one grid cell once per seed and write the whole folder.

    build_model is a zero-argument factory returning an un-moved nn.Module; it is called once per
    seed, after set_seed, so the initialisation it draws is the seed's.
    """
    enable_cublas_determinism()
    configure_determinism(config.deterministic)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # the decode is RNG-free, so the splits are loaded once and shared by every seed
    splits = tuple(
        GPUDataset(
            f"{config.data_dir}/{name}.csv",
            device,
            normalization=config.normalization,
            geometry=config.geometry,
        )
        for name in ("train", "val", "test")
    )
    train_data, val_data, test_data = splits

    n_pos = train_data.labels.sum()
    class_weight = (len(train_data) - n_pos) / n_pos

    out_dir = config.out_dir()
    # rewritten wholesale: a shorter seeds list must not leave the previous run's seed folders
    # sitting next to the new ones
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": config.model_name,
                "data_dir": config.data_dir,
                "normalization": config.normalization,
                "geometry": config.geometry,
                "class_weighting": config.cw,
                "class_weight": float(class_weight),
                "seeds": list(config.seeds),
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "deterministic": config.deterministic,
                "device": device,
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
            },
            indent=2,
        )
    )

    results = []
    for seed in config.seeds:
        print(f"\n=== Seed {seed} ({config.model_name}, {config.data_dir}, CW={config.cw}) ===")
        results.append(
            (seed, run_seed(build_model, seed, splits, class_weight, config, out_dir / f"seed_{seed}"))
        )

    aggregate(results, out_dir, config)
    print(f"\nWrote {len(results)} seed(s) to {out_dir}")
    return out_dir
