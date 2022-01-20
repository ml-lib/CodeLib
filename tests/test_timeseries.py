"""
Test suite module for ``timeseries``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Jan 05, 2022
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
    """Test suite for module ``TimeSeries``."""

    def setUp(self):
        """Set up for module ``TimeSeries``."""

    def test_multivariate(self):
        """TimeSeries: Test for multivariate"""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip = df_ip.set_index("ts")
        y_var = "y"
        x_var = ["cost"]
        mod = AutoArima(df=df_ip, y_var=y_var, x_var=x_var)
        metrics = mod.model_summary
        X = pd.DataFrame(df_ip.iloc[-1]).T
        op = mod.predict(x_predict=X[x_var])[y_var][0]
        exp_op = X[y_var][0]
        self.assertEqual(mod.opt_param["order"], (0, 1, 1))
        self.assertAlmostEqual(1.0, metrics["rsq"], places=1)
        self.assertLessEqual(metrics["mape"], 0.1)
        self.assertAlmostEqual(op, exp_op, places=0)

    def test_univariate(self):
        """TimeSeries: Test for univariate"""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="endog")
        df_ip = df_ip.set_index("ts")
        mod = AutoArima(df=df_ip, y_var="Passengers")
        op = mod.predict()
        self.assertAlmostEqual(op["Passengers"].values[0], 445.634, places=1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
