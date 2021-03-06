"""
Test suite module for ``gam``.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 31, 2022
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

from mllib.lib.gam import GAM  # noqa: F841

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


class TestGAM(unittest.TestCase):
    """Test suite for module ``gam``."""

    def setUp(self):
        """Set up for module ``gam``."""

    @ignore_warnings
    def test_gam_exog(self):
        """GAM: Test with exogenous variable"""
        x_var = ["day_of_week", "cp", "stock_level", "retail_price"]
        y_var = "y"
        test_perc = 0.2
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip["day_of_week"] = df_ip['ts'].dt.weekday
        df_train = df_ip.iloc[0:int(len(df_ip) * (1-test_perc)), :]
        df_test = df_ip.iloc[int(len(df_ip) * (1-test_perc)): len(df_ip), :]
        df_test = df_test[x_var]
        mod = GAM(df_train, y_var, x_var, splines=100)
        mod.predict(df_test)
        metrics = mod.model_summary
        self.assertGreaterEqual(metrics["rsq"], 0.9)
        self.assertLessEqual(metrics["mape"], 0.4)

    @ignore_warnings
    def test_gam_linear_cubic_variable(self):
        """GAM: Test for linear and cubic order fit"""
        x_var = ["day_of_week", "cp", "stock_level", "retail_price"]
        y_var = "y"
        linear_var = ['stock_level']
        cubic_var = ['retail_price']
        test_perc = 0.2
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip["day_of_week"] = df_ip['ts'].dt.weekday
        df_train = df_ip.iloc[0:int(len(df_ip) * (1-test_perc)), :]
        df_test = df_ip.iloc[int(len(df_ip) * (1-test_perc)): len(df_ip), :]
        df_test = df_test[x_var]
        mod = GAM(df_train, y_var, x_var, linear_var, cubic_var, splines=100)
        mod.predict(df_test)
        metrics = mod.model_summary
        self.assertGreaterEqual(metrics["rsq"], 0.9)
        self.assertLessEqual(metrics["mape"], 0.4)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
