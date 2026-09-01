import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display needed, we only ever write files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# mcc is the only metric that runs negative; everything else is a rate in [0, 1]
METRIC_LIMITS = {"mcc": (-1, 1)}

# the metrics a summary bar chart shows; counts (tn/fp/fn/tp) and loss are on other scales
SUMMARY_METRICS = (
    "accuracy",
    "auroc",
    "auprc",
    "auprc_normal",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1",
)


def _ensure_parent(save_path: str) -> None:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


class MetricTracker:
    """Collects per-epoch metrics for each phase and plots the curves."""

    def __init__(self):
        self.history = {}

    def record(self, phase: str, epoch: int, loss: float, metrics: dict = None):
        """Store one epoch of results.

        metrics is a dict from evaluation_metrics.compute_metrics; every scalar in it is
        tracked and the rest (the confusion matrix) is skipped.
        """
        entry = self.history.setdefault(phase, {"epoch": [], "loss": []})
        entry["epoch"].append(epoch)
        entry["loss"].append(loss)

        for name, value in (metrics or {}).items():
            if name == "loss" or not isinstance(value, (int, float)):
                continue
            entry.setdefault(name, []).append(value)

    def to_csv(self, save_path: str):
        """The whole history as one wide CSV: a row per (phase, epoch), a column per metric.

        Written per seed so the curves behind a plot are still readable as numbers -- the paper
        needs the raw values, not only the picture.
        """
        rows = []
        for phase, entry in self.history.items():
            for position, epoch in enumerate(entry["epoch"]):
                row = {"phase": phase, "epoch": epoch}
                for name, values in entry.items():
                    # a metric can be shorter than "epoch" if it only started being recorded
                    # later, so index rather than zip the whole column
                    if name != "epoch" and position < len(values):
                        row[name] = values[position]
                rows.append(row)

        _ensure_parent(save_path)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        return save_path

    def plot(self, save_path: str = "reports/curves.png", metrics=("accuracy",)):
        if not self.history:
            raise ValueError("nothing to plot, record() was never called")

        # a metric nobody recorded gets dropped rather than drawn as an empty panel
        names = ["loss"] + [m for m in metrics if any(m in e for e in self.history.values())]
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
        axes = axes[0]

        for name, ax in zip(names, axes):
            for phase, entry in self.history.items():
                if name not in entry:
                    continue
                # a marker keeps a single-point phase (e.g. a final-only test run) visible
                ax.plot(entry["epoch"][: len(entry[name])], entry[name], marker="o", label=phase)

            ax.set(xlabel="epoch", ylabel=name, title=name.replace("_", " ").title())
            if name != "loss":
                ax.set_ylim(*METRIC_LIMITS.get(name, (0, 1)))

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)
            ax.legend()

        fig.tight_layout()
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved curves to {save_path}")


def plot_mean_curves(histories, save_path: str, metrics=("accuracy",)):
    """Mean +/- SD curves across the seeds of one cell.

    histories is a list of MetricTracker.history dicts, one per seed. Early stopping makes the
    runs different lengths, so every phase is truncated to its shortest run: past that point the
    mean would be over a changing subset of seeds and the band would silently narrow for a
    reason that has nothing to do with the model.
    """
    histories = [history for history in histories if history]
    if not histories:
        raise ValueError("nothing to plot, no histories were recorded")

    phases = [phase for phase in histories[0] if all(phase in h for h in histories)]
    names = ["loss"] + [
        m for m in metrics if any(m in h[phase] for h in histories for phase in phases)
    ]

    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    axes = axes[0]

    for name, ax in zip(names, axes):
        for phase in phases:
            columns = [h[phase][name] for h in histories if name in h[phase]]
            if not columns:
                continue

            length = min(len(column) for column in columns)
            stacked = np.array([column[:length] for column in columns], dtype=float)
            mean = stacked.mean(axis=0)
            # ddof=1 is the sample SD -- these 5 seeds stand in for the population of seeds.
            # A single run has no spread to draw, so the band collapses to the line.
            sd = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros(length)
            epochs = histories[0][phase]["epoch"][:length]

            line, = ax.plot(epochs, mean, marker="o", label=f"{phase} (n={len(stacked)})")
            ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.2, color=line.get_color())

        ax.set(xlabel="epoch", ylabel=name, title=f"{name.replace('_', ' ').title()} (mean +/- SD)")
        if name != "loss":
            ax.set_ylim(*METRIC_LIMITS.get(name, (0, 1)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved mean curves to {save_path}")
    return save_path


def summarize_seed_metrics(rows, split: str = None):
    """One row per seed -> one row per metric, with mean and sample SD.

    rows is a list of the per-seed metric dicts (or a DataFrame of them). "seed" is carried
    through as an index column rather than averaged. split, if given, is written into a column
    so the test and validation summaries can share one file.
    """
    frame = pd.DataFrame(rows)
    values = frame.drop(columns=[c for c in ("seed", "split") if c in frame], errors="ignore")
    values = values.select_dtypes("number")

    summary = pd.DataFrame(
        {
            "metric": values.columns,
            "mean": values.mean().to_numpy(),
            # ddof=1 throughout; pandas already defaults to it, stated here because the
            # per-seed spread is the number the paper reports
            "sd": values.std(ddof=1).to_numpy(),
            "n": values.count().to_numpy(),
        }
    )

    if split is not None:
        summary.insert(0, "split", split)

    return summary


def plot_metric_summary(summary, save_path: str, title: str = "Test metrics (mean +/- SD)"):
    """Bar chart of the summarised metrics with the seed SD as error bars."""
    frame = summary[summary["metric"].isin(SUMMARY_METRICS)]
    # keep SUMMARY_METRICS' order rather than whatever order the columns arrived in
    frame = frame.set_index("metric").reindex([m for m in SUMMARY_METRICS if m in set(frame["metric"])])

    fig, ax = plt.subplots(figsize=(1.1 * len(frame) + 3, 5))
    positions = np.arange(len(frame))
    sd = frame["sd"].fillna(0.0).to_numpy()
    ax.bar(positions, frame["mean"], yerr=sd, capsize=4, color="#4c72b0", alpha=0.85)

    for position, mean, spread in zip(positions, frame["mean"], sd):
        ax.text(position, mean + spread + 0.02, f"{mean:.3f}", ha="center", fontsize=8)

    lowest = float(min(0.0, (frame["mean"] - sd).min()))
    ax.set(
        ylabel="score",
        title=title,
        xticks=positions,
        ylim=(lowest - 0.05, 1.1),
    )
    ax.set_xticklabels([name.replace("_", "\n") for name in frame.index], fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved metric summary to {save_path}")
    return save_path


def format_summary(summary, title: str) -> str:
    """The mean +/- SD table as text, for the summary.txt a reader opens first."""
    lines = [title, "-" * len(title)]
    for _, row in summary.iterrows():
        # counts are whole images, the rest are rates; a single seed has no spread to report
        digits = 1 if row["metric"] in ("tn", "fp", "fn", "tp") else 4
        spread = "n/a" if pd.isna(row["sd"]) else f"{row['sd']:.{digits}f}"
        lines.append(f"  {row['metric']:<20} {row['mean']:>10.{digits}f} +/- {spread}")
    return "\n".join(lines)
