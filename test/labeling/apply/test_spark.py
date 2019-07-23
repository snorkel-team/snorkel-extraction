import unittest

import numpy as np
import pandas as pd
import pytest
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext

from snorkel.labeling.apply.spark import SparkLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: Row) -> Row:
    return Row(num=x.num, num_squared=x.num ** 2)


@labeling_function()
def f(x):
    if x.num > 42:
        return 0
    return None


@labeling_function(preprocessors=[square])
def fp(x):
    if x.num_squared > 42:
        return 0
    return None


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x, db):
    if x.num in db:
        return 0
    return None


DATA = [3, 43, 12, 9, 3]
L_EXPECTED = np.array([[-1, 0], [0, -1], [-1, -1], [-1, 0], [-1, 0]])
L_PREPROCESS_EXPECTED = np.array([[-1, -1], [0, 0], [-1, 0], [-1, 0], [-1, -1]])

TEXT_DATA = ["Jane", "Jane plays soccer.", "Jane plays soccer."]
L_TEXT_EXPECTED = np.array([[0, -1], [0, 0], [0, 0]])


class TestSparkApplier(unittest.TestCase):
    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, g])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_preprocessor(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, fp])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_pandas_preprocessor_memoized(self) -> None:
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)

        @preprocessor(memoize=True)
        def square_memoize(x: DataPoint) -> DataPoint:
            return Row(num=x.num, num_squared=x.num ** 2)

        @labeling_function(preprocessors=[square_memoize])
        def fp_memoized(x):
            if x.num_squared > 42:
                return 0
            return None

        df = pd.DataFrame(dict(num=DATA))
        rdd = sql.createDataFrame(df).rdd
        applier = SparkLFApplier([f, fp_memoized])
        L = applier.apply(rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)
