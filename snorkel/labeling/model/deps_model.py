from typing import Any, Dict, List, NamedTuple, Optional, Set, Union

import numpy as np
import scipy.sparse as sparse

from snorkel.labeling.model import LabelModel

Matrix = Union[np.ndarray, sparse.spmatrix]
TrainConfig = Dict[str, Any]
Metrics = Dict[str, float]


class DependencyLearner(LabelModel):
    """A DependencyLearner to learn pairwise LF dependencies.

    Parameters
    ----------
    cardinality
        Number of classes, by default 2
        TODO: only works for the binary case right now
    **kwargs
        Arguments for changing config defaults

    Attributes
    ----------
    cardinality
        Number of classes, by default 2

    Examples
    --------
    ```
    dep_learner = DependencyLearner()
    dep_learner = DependencyLearner(cardinality=3)
    ```
    """

    def __init__(self, cardinality: int = 2) -> None:
        super().__init__()
        self.cardinality = cardinality

    def _check_L(self, L: Matrix) -> np.ndarray:
        """
        Check label matrix format and content. Convert to dense matrix if needed.

        Parameters
        ----------
        L
            A [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, m] dense matrix of labels

        Raises
        ------
        ValueError
            If values in L are less than 0
        """
        if sparse.issparse(L):
            L = L.todense()

        # Check for correct values, e.g. warning if in {-1,0,1}
        if np.any(L < 0):
            raise ValueError("L must have values in {0,1,...,k}.")

        return L

    def _set_constants(self, L: Matrix) -> None:
        self.n, self.m = L.shape

    def _generate_O(self, L: Matrix) -> None:
        """Generate overlaps and conflicts matrix from label matrix.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        """
        


        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy(L_aug.T @ L_aug / self.n).float()

    def learn_deps(self, L_train: Matrix) -> None:
        L_train = self._check_L(L_train)
        self._set_constants(L_train)
        self._generate_O(L_train)

