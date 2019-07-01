import unittest

import numpy as np
import torch

from snorkel.analysis.metrics import metric_score


class MetricsTest(unittest.TestCase):
    def test_accuracy_basic(self):
        golds = [1, 1, 1, 2, 2]
        preds = [1, 1, 1, 2, 1]
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.8)

    def test_bad_inputs(self):
        golds = [1, 1, 1, 2, 2]
        pred1 = [1, 1, 1, 2, 0.5]
        pred2 = "1 1 1 2 2"
        pred3 = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        self.assertRaises(
            ValueError, metric_score, golds, pred1, probs=None, metric="accuracy"
        )
        self.assertRaises(
            ValueError, metric_score, golds, pred2, probs=None, metric="accuracy"
        )
        self.assertRaises(
            ValueError, metric_score, golds, pred3, probs=None, metric="accuracy"
        )

    def test_array_conversion(self):
        golds = torch.Tensor([1, 1, 1, 2, 2])
        preds = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.8)

    def test_ignores(self):
        golds = [1, 1, 1, 2, 2]
        preds = [1, 0, 1, 2, 1]
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.6)
        score = metric_score(
            golds, preds, probs=None, metric="accuracy", ignore_in_preds=[0]
        )
        self.assertAlmostEqual(score, 0.75)
        score = metric_score(
            golds, preds, probs=None, metric="accuracy", ignore_in_golds=[1]
        )
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(
            golds,
            preds,
            probs=None,
            metric="accuracy",
            ignore_in_golds=[2],
            ignore_in_preds=[0],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_coverage(self):
        golds = [1, 1, 1, 1, 2]
        preds = [0, 0, 1, 1, 1]
        score = metric_score(golds, preds, probs=None, metric="coverage")
        self.assertAlmostEqual(score, 0.6)
        score = metric_score(
            golds, preds, probs=None, ignore_in_golds=[2], metric="coverage"
        )
        self.assertAlmostEqual(score, 0.5)

    def test_precision(self):
        golds = [1, 1, 1, 2, 2]
        preds = [2, 2, 1, 1, 2]
        score = metric_score(golds, preds, probs=None, metric="precision")
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(golds, preds, probs=None, metric="precision", pos_label=2)
        self.assertAlmostEqual(score, 0.333, places=2)

    def test_recall(self):
        golds = [1, 1, 1, 1, 2]
        preds = [2, 2, 1, 1, 2]
        score = metric_score(golds, preds, probs=None, metric="recall")
        self.assertAlmostEqual(score, 0.5)
        score = metric_score(golds, preds, probs=None, metric="recall", pos_label=2)
        self.assertAlmostEqual(score, 1.0)

    def test_f1(self):
        golds = [1, 1, 1, 1, 2]
        preds = [2, 2, 1, 1, 2]
        score = metric_score(golds, preds, probs=None, metric="f1")
        self.assertAlmostEqual(score, 0.666, places=2)
        score = metric_score(golds, preds, probs=None, pos_label=2, metric="f1")
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_fbeta(self):
        golds = [1, 1, 1, 1, 2]
        preds = [2, 2, 1, 1, 2]
        pre = metric_score(golds, preds, probs=None, metric="precision")
        rec = metric_score(golds, preds, probs=None, metric="recall")
        self.assertAlmostEqual(
            pre,
            metric_score(golds, preds, probs=None, metric="fbeta", beta=1e-6),
            places=2,
        )
        self.assertAlmostEqual(
            rec,
            metric_score(golds, preds, probs=None, metric="fbeta", beta=1e6),
            places=2,
        )

    def test_matthews(self):
        golds = [1, 1, 1, 1, 2]
        preds = [2, 1, 1, 1, 1]
        mcc = metric_score(golds, preds, probs=None, metric="matthews_corrcoef")
        self.assertAlmostEqual(mcc, -0.25)

        golds = [1, 1, 1, 1, 2]
        preds = [1, 1, 1, 1, 2]
        mcc = metric_score(golds, preds, probs=None, metric="matthews_corrcoef")
        self.assertAlmostEqual(mcc, 1.0)

    def test_roc_auc(self):
        golds = [1, 1, 1, 1, 2]
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        roc_auc = metric_score(golds, preds=None, probs=probs, metric="roc_auc")
        self.assertAlmostEqual(roc_auc, 0.0)
        probs = np.fliplr(probs)
        roc_auc = metric_score(golds, preds=None, probs=probs, metric="roc_auc")
        self.assertAlmostEqual(roc_auc, 1.0)


if __name__ == "__main__":
    unittest.main()
