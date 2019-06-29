from typing import Callable, List, NamedTuple

import numpy as np
import sklearn.metrics as skmetrics

from .utils import filter_labels


class Metric(NamedTuple):
    func: Callable[..., float]
    inputs: List[str] = ["golds", "preds"]


def metric_score(
    golds: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric: str,
    ignore_in_golds: List[int] = [],
    ignore_in_preds: List[int] = [],
    **kwargs,
):
    if metric not in METRICS:
        msg = f"The metric you provided ({metric}) is not currently implemented."
        raise ValueError(msg)

    # Optionally filter out examples (e.g., abstain predictions or unknown labels)
    golds, preds, probs = filter_labels(
        golds, preds, probs, ignore_in_golds, ignore_in_preds
    )

    # Pass the metric function its requested args
    func, input_names = METRICS[metric]
    input_map = {"golds": golds, "preds": preds, "probs": probs}
    inputs = [input_map[input] for input in input_names]

    return func(*inputs, **kwargs)


def coverage_score(preds):
    return np.sum(preds != 0) / len(preds)


# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# for details on the definitions and available kwargs for all metrics from scikit-learn
METRICS = {
    "accuracy": Metric(skmetrics.accuracy_score),
    "coverage": Metric(coverage_score, ["preds"]),
    "f1": Metric(skmetrics.f1_score),
    "fbeta": Metric(skmetrics.fbeta_score),
    "matthews_corrcoef": Metric(skmetrics.matthews_corrcoef),
    "precision": Metric(skmetrics.precision_score),
    "recall": Metric(skmetrics.recall_score),
    "roc_auc": Metric(skmetrics.roc_auc_score, ["golds", "probs"]),
}
