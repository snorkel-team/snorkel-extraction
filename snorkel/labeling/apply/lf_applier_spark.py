from typing import Tuple

# NB: don't include pyspark in requirements.txt to avoid
# overwriting existing system Spark install
import scipy.sparse as sparse
from pyspark import RDD

from snorkel.labeling.preprocess import PreprocessorMode
from snorkel.types import DataPoint

from .lf_applier import BaseLFApplier, RowData, apply_lfs_to_data_point


class SparkLFApplier(BaseLFApplier):
    """LF applier for a Spark RDD.

    Data points are stored as `Row`s in an RDD, and a Spark
    `map` job is submitted to execute the LFs. A common
    way to obtain an RDD is via a PySpark DataFrame. For an
    example usage with AWS EMR instructions, see
    `test/labeling/apply/lf_applier_spark_test_script.py`.
    """

    def apply(self, data_points: RDD) -> sparse.csr_matrix:  # type: ignore
        """Label PySpark RDD of data points with LFs.

        Parameters
        ----------
        data_points
            PySpark RDD containing data points to be labeled by LFs

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of labels emitted by LFs
        """

        def map_fn(args: Tuple[DataPoint, int]) -> RowData:
            return apply_lfs_to_data_point(*args, lfs=self._lfs)

        self._set_lf_preprocessor_mode(PreprocessorMode.SPARK)
        labels = data_points.zipWithIndex().map(map_fn).collect()
        return self._matrix_from_row_data(labels)
