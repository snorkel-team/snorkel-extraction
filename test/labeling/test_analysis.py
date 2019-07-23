import unittest

import numpy as np
import pandas as pd

from snorkel.labeling.analysis import LFAnalysis

L = [
    [-1, -1, 0, -1, -1, 0],
    [-1, -1, -1, 2, -1, -1],
    [2, -1, -1, -1, -1, 0],
    [1, -1, 2, -1, 0, 0],
    [-1, -1, -1, -1, -1, -1],
    [1, -1, 0, 2, 1, 0],
]

Y = [0, 1, 2, 0, 1, 2]


class TestAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.LFA = LFAnalysis(np.array(L))
        self.Y = np.array(Y)

    def test_label_coverage(self) -> None:
        self.assertEqual(self.LFA.label_coverage(), 5 / 6)

    def test_label_overlap(self) -> None:
        self.assertEqual(self.LFA.label_overlap(), 4 / 6)

    def test_label_conflict(self) -> None:
        self.assertEqual(self.LFA.label_conflict(), 3 / 6)

    def test_lf_polarities(self) -> None:
        polarities = self.LFA.lf_polarities()
        self.assertEqual(polarities, [[1, 2], [], [0, 2], [2], [0, 1], [0]])

    def test_lf_coverages(self) -> None:
        coverages = self.LFA.lf_coverages()
        coverages_expected = [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6]
        np.testing.assert_array_almost_equal(coverages, np.array(coverages_expected))

    def test_lf_overlaps(self) -> None:
        overlaps = self.LFA.lf_overlaps(normalize_by_coverage=False)
        overlaps_expected = [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6]
        np.testing.assert_array_almost_equal(overlaps, np.array(overlaps_expected))

        overlaps = self.LFA.lf_overlaps(normalize_by_coverage=True)
        overlaps_expected = [1, 0, 1, 1 / 2, 1, 1]
        np.testing.assert_array_almost_equal(overlaps, np.array(overlaps_expected))

    def test_lf_conflicts(self) -> None:
        conflicts = self.LFA.lf_conflicts(normalize_by_overlaps=False)
        conflicts_expected = [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6]
        np.testing.assert_array_almost_equal(conflicts, np.array(conflicts_expected))

        conflicts = self.LFA.lf_conflicts(normalize_by_overlaps=True)
        conflicts_expected = [1, 0, 2 / 3, 1, 1, 3 / 4]
        np.testing.assert_array_almost_equal(conflicts, np.array(conflicts_expected))

    def test_lf_empirical_accuracies(self) -> None:
        accs = self.LFA.lf_empirical_accuracies(self.Y)
        accs_expected = [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4]
        np.testing.assert_array_almost_equal(accs, np.array(accs_expected))

    def test_lf_empirical_probs(self) -> None:
        P_emp = self.LFA.lf_empirical_probs(self.Y, 3)
        P = np.array(
            [
                [[1 / 2, 1, 0], [0, 0, 0], [1 / 2, 0, 1 / 2], [0, 0, 1 / 2]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 1 / 2], [1 / 2, 0, 1 / 2], [0, 0, 0], [1 / 2, 0, 0]],
                [[1, 1 / 2, 1 / 2], [0, 0, 0], [0, 0, 0], [0, 1 / 2, 1 / 2]],
                [[1 / 2, 1, 1 / 2], [1 / 2, 0, 0], [0, 0, 1 / 2], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]],
            ]
        )
        np.testing.assert_array_almost_equal(P, P_emp)

    def test_lf_summary(self) -> None:
        df = self.LFA.lf_summary(self.Y, lf_names=None, est_accs=None)
        df_expected = pd.DataFrame(
            {
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
                "Correct": [1, 0, 1, 1, 1, 2],
                "Incorrect": [2, 0, 2, 1, 1, 2],
                "Emp. Acc.": [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4],
            }
        )
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))

        df = self.LFA.lf_summary(Y=None, lf_names=None, est_accs=None)
        df_expected = pd.DataFrame(
            {
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
            }
        )
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))

        est_accs = [1, 0, 1, 1, 1, 0.5]
        names = list("abcdef")
        df = self.LFA.lf_summary(self.Y, lf_names=names, est_accs=est_accs)
        df_expected = pd.DataFrame(
            {
                "j": [0, 1, 2, 3, 4, 5],
                "Polarity": [[1, 2], [], [0, 2], [2], [0, 1], [0]],
                "Coverage": [3 / 6, 0, 3 / 6, 2 / 6, 2 / 6, 4 / 6],
                "Overlaps": [3 / 6, 0, 3 / 6, 1 / 6, 2 / 6, 4 / 6],
                "Conflicts": [3 / 6, 0, 2 / 6, 1 / 6, 2 / 6, 3 / 6],
                "Correct": [1, 0, 1, 1, 1, 2],
                "Incorrect": [2, 0, 2, 1, 1, 2],
                "Emp. Acc.": [1 / 3, 0, 1 / 3, 1 / 2, 1 / 2, 2 / 4],
                "Learned Acc.": [1, 0, 1, 1, 1, 0.5],
            }
        ).set_index(pd.Index(names))
        pd.testing.assert_frame_equal(df.round(6), df_expected.round(6))
