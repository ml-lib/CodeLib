"""
Test suite module for ``prophet``.
Credits
-------
::
    Authors:
        - Diptesh
    Date: Sep 07, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import pandas as pd

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.prophet import FBP  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestFBP(unittest.TestCase):
    """Test suite for module ``prophet``."""

    def setUp(self):
        """Set up for module ``prophet``."""

    @ignore_warnings
    def test_multivariate(self):
        """TimeSeries: Test for multivariate."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        param = {"interval_width": [0.95],
                 "changepoint_prior_scale": [0.1, 0.5],
                 "seasonality_prior_scale": [0.01, 10.0]}
        mod = FBP(df=df_ip,
                  y_var="y",
                  x_var=["cost", "stock_level", "retail_price"],
                  ds="ts",
                  param = param)
        op = mod.model_summary
        self.assertAlmostEqual(0.99, op["rsq"], places=1)

    @ignore_warnings
    def test_raise_exceptions(self):
        """TimeSeries: Test raise exceptions."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        self.assertRaises(NotImplementedError, FBP,
                          df=df_ip,
                          y_var="y",
                          x_var=["stock_level", "retail_price"],
                          ds="ts",
                          uid="cost")
        self.assertRaises(NotImplementedError, FBP,
                          df=df_ip,
                          y_var="y",
                          x_var=["stock_level", "retail_price"],
                          ds="ts",
                          k_fold=5)

    # @ignore_warnings
    # def test_univariate(self):
    #     """TimeSeries: Test for univariate."""
    #     df_ip = pd.read_excel(path + "test_time_series.xlsx",
    #                           sheet_name="endog")
    #     param = {"interval_width": [0.95],
    #              "changepoint_prior_scale": [0.001],
    #              "seasonality_prior_scale": [0.01, 10.0]}
    #     mod = FBP(df=df_ip, y_var="Passengers", ds="ts", param = param)
    #     op = mod.predict()
    #     self.assertAlmostEqual(op["y"].values[0], 446.911, places=1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
