"""Binary classification metrics for the pneumonia task.

Labels follow preprocessing.LABEL_MAP: NORMAL=0, PNEUMONIA=1, so the positive class is
PNEUMONIA. The models emit raw logits, so the default decision threshold is 0 (p > 0.5).

The imbalance strategies this paper compares mostly move the decision boundary rather than
change how well the model separates the classes, so the threshold-free scores (auroc, auprc)
are the ones that say whether a method actually helped.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# NORMAL is the minority class here (1214 vs 3494 in train), but PNEUMONIA is the clinically
# positive one, so the two conventions pull apart. Positive = PNEUMONIA throughout, with
# auprc_normal covering the minority side.
NEGATIVE_LABEL = "NORMAL"
POSITIVE_LABEL = "PNEUMONIA"

# the order the scalar keys are printed in
_SCALAR_KEYS = (
    "loss",
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


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _ratio(numerator, denominator) -> float:
    """A rate that is undefined rather than 0 when its denominator is empty."""
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def compute_metrics(logits, labels, threshold: float = 0.0, loss: float = None) -> dict:
    """Every metric for one set of predictions.

    logits are raw model outputs (pre-sigmoid) and labels are 0/1; both accept torch tensors
    on any device or anything array-like. threshold is on the logit scale, so the default of 0
    is the usual p > 0.5 cut -- pass a validation-tuned value to score a different operating
    point. loss, if given, is carried through into the result unchanged.
    """
    logits = _to_numpy(logits).ravel()
    labels = _to_numpy(labels).ravel().astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    predictions = (logits > threshold).astype(int)

    # labels=[0, 1] keeps the matrix 2x2 even when a class is missing from this set
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)

    metrics = {
        "accuracy": _ratio(tp + tn, len(labels)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix,
    }

    # the ranking metrics are undefined on a single class, and sklearn raises there
    if len(np.unique(labels)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        metrics["auprc_normal"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
        # the minority-class view: NORMAL as positive, so its baseline is NORMAL prevalence
        metrics["auprc_normal"] = float(average_precision_score(1 - labels, 1.0 - probabilities))

    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics


class PredictionCollector:
    """Gathers a whole split's predictions before scoring it.

    auroc and auprc need every prediction at once, so unlike a running accuracy they cannot be
    accumulated batch by batch. Batches are kept as-is on the device and concatenated once in
    compute(), which keeps this to a single host sync per epoch.
    """

    def __init__(self):
        self.logits = []
        self.labels = []
        self.loss_total = None
        self.count = 0

    def update(self, logits, labels, loss=None):
        """Add one batch. loss is the batch mean, weighted here by the batch size."""
        self.logits.append(logits.detach())
        self.labels.append(labels.detach())
        self.count += logits.size(0)

        if loss is not None:
            weighted = loss.detach() * logits.size(0)
            self.loss_total = weighted if self.loss_total is None else self.loss_total + weighted

    def compute(self, threshold: float = 0.0) -> dict:
        if not self.logits:
            raise ValueError("nothing to compute, update() was never called")

        loss = None if self.loss_total is None else self.loss_total.item() / self.count
        return compute_metrics(
            torch.cat(self.logits),
            torch.cat(self.labels),
            threshold=threshold,
            loss=loss,
        )


def format_metrics(metrics: dict, phase: str) -> str:
    """The per-epoch report: scalars on two lines, then the confusion matrix."""
    scalars = {key: metrics[key] for key in _SCALAR_KEYS if key in metrics}

    def line(keys, labels):
        parts = [f"{label} {scalars[key]:.4f}" for key, label in zip(keys, labels) if key in scalars]
        return "  " + " | ".join(parts)

    lines = [f"{phase}:"]
    lines.append(
        line(
            ("loss", "accuracy", "auroc", "auprc", "auprc_normal"),
            ("loss", "acc", "auroc", "auprc", "auprc(N)"),
        )
    )
    lines.append(
        line(
            ("sensitivity", "specificity", "balanced_accuracy", "mcc", "f1"),
            ("sens", "spec", "bal_acc", "mcc", "f1"),
        )
    )

    matrix = metrics.get("confusion_matrix")
    if matrix is not None:
        width = max(len(NEGATIVE_LABEL), len(POSITIVE_LABEL))
        lines.append("  Confusion matrix (rows = true):")
        lines.append(f"    {'':{width}}  {'pred ' + NEGATIVE_LABEL:>14}  {'pred ' + POSITIVE_LABEL:>14}")
        for name, row in zip((NEGATIVE_LABEL, POSITIVE_LABEL), matrix):
            lines.append(f"    {name:{width}}  {row[0]:>14}  {row[1]:>14}")

    return "\n".join(lines)
