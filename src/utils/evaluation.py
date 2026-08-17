import os

import matplotlib

matplotlib.use("Agg")  # no display needed, we only ever write files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# mcc is the only metric that runs negative; everything else is a rate in [0, 1]
METRIC_LIMITS = {"mcc": (-1, 1)}


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
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved curves to {save_path}")
