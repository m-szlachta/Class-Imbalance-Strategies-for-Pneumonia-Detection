import os

import matplotlib

matplotlib.use("Agg")  # no display needed, we only ever write files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class MetricTracker:
    """Collects per-epoch loss and accuracy for each phase and plots the curves."""

    def __init__(self):
        self.history = {}

    def record(self, phase: str, epoch: int, loss: float, accuracy: float):
        """Store one epoch of results. accuracy is a fraction in [0, 1]."""
        entry = self.history.setdefault(phase, {"epoch": [], "loss": [], "accuracy": []})
        entry["epoch"].append(epoch)
        entry["loss"].append(loss)
        entry["accuracy"].append(accuracy)

    def plot(self, save_path: str = "reports/curves.png"):
        if not self.history:
            raise ValueError("nothing to plot, record() was never called")

        fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 5))

        for phase, entry in self.history.items():
            # a marker keeps a single-point phase (e.g. a final-only test run) visible
            loss_ax.plot(entry["epoch"], entry["loss"], marker="o", label=phase)
            acc_ax.plot(
                entry["epoch"],
                [a * 100 for a in entry["accuracy"]],
                marker="o",
                label=phase,
            )

        loss_ax.set(xlabel="epoch", ylabel="loss", title="Loss")
        acc_ax.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy", ylim=(0, 100))
        for ax in (loss_ax, acc_ax):
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
