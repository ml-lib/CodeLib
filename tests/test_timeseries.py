"""
Test suite module for ``timeseries``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Feb 17, 2024
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=W0511,W0611

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import xlrd
import openpyxl

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.timeseries import AutoArima  # noqa: F841
from mllib.lib.timeseries import BatesGrager  # noqa: F841

__all__ = ["xlrd", "openpyxl", ]

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


# TODO: Change integration tests.
class TestTimeSeries(unittest.TestCase):
    """Test suite for module ``timeseries``."""

    def setUp(self):
        """Set up for module ``timeseries``."""

    def test_multivariate(self):
        """AutoArima: Test for multivariate"""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip = df_ip.set_index("ts")
        y_var = "y"
        x_var = ["cost"]
        mod = AutoArima(df=df_ip, y_var=y_var, x_var=x_var)
        metrics = mod.model_summary
        X = pd.DataFrame(df_ip.iloc[-1]).T
        op = mod.predict(x_predict=X[x_var])[y_var].iloc[0]
        exp_op = X[y_var][0]
        self.assertEqual(mod.opt_param["order"], (0, 1, 1))
        self.assertAlmostEqual(1.0, metrics["rsq"], places=1)
        self.assertLessEqual(metrics["mape"], 0.1)
        self.assertAlmostEqual(op, exp_op, places=0)

    def test_univariate(self):
        """AutoArima: Test for univariate"""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="endog")
        df_ip = df_ip.set_index("ts")
        mod = AutoArima(df=df_ip, y_var="Passengers")
        op = mod.predict()
        self.assertAlmostEqual(op["Passengers"].values[0], 445.634, places=1)

    def test_bates_granger(self):
        """TimesSeries: Test for Bates & Granger"""
        df_raw = pd.read_excel(path + "test_time_series.xlsx",
                               sheet_name="bates_granger")
        exp_op = df_raw[["ts", "y", "y_hat_01", "y_hat_02",
                         "y_hat_03", "y_hat_04", "y_hat_bg"]].fillna(0)
        df_ip = exp_op.drop("y_hat_bg", axis=1)
        mod = BatesGrager(df=df_ip,
                          y="y",
                          y_hat=["y_hat_01", "y_hat_02",
                                 "y_hat_03", "y_hat_04"],
                          lag=53, pred_period=1)
        op = mod.solve()
        pd.testing.assert_frame_equal(op, exp_op)

    def test_bates_granger_infeasible(self):
        """TimesSeries: Test for Bates & Granger infeasibility"""
        df_raw = pd.read_excel(path + "test_time_series.xlsx",
                               sheet_name="bates_granger")
        exp_op = df_raw[["ts", "y", "y_hat_01", "y_hat_02",
                         "y_hat_03", "y_hat_04", "y_hat_bg"]].fillna(0)
        df_ip = exp_op.drop("y_hat_bg", axis=1)
        mod = BatesGrager(df=df_ip,
                          y="y",
                          y_hat=["y_hat_01", "y_hat_02",
                                 "y_hat_03", "y_hat_04"],
                          lag=100, pred_period=10)
        with self.assertRaises(AssertionError):
            mod.solve()


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
