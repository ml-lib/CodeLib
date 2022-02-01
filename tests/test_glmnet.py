"""
Test suite module for ``glmnet``.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 28, 2022
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
import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.glmnet import GLMNet  # noqa: F841

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


class TestGLMNet(unittest.TestCase):
    """Test suite for module ``GLMNet``."""

    def setUp(self):
        """Set up for module ``GLMNet``."""

    def test_known_equation(self):
        """GLMNet: Test a known equation"""
        df_ip = pd.read_csv(path + "test_glmnet.csv")
        mod = GLMNet(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
        op = mod.best_params_
        self.assertEqual(np.round(op.get("intercept"), 0), 100.0)
        self.assertEqual(np.round(op.get("coef")[0], 0), 2.0)
        self.assertEqual(np.round(op.get("coef")[1], 0), 3.0)
        self.assertEqual(np.round(op.get("coef")[2], 0), 0.0)

    def test_predict_target_variable(self):
        """GLMNet: Test to predict a target variable"""
        df_ip = pd.read_csv(path + "test_glmnet.csv")
        mod = GLMNet(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
        df_predict = pd.DataFrame({"x1": [10, 20],
                                   "x2": [5, 10],
                                   "x3": [100, 0]})
        op = mod.predict(df_predict)
        op = np.round(np.array(op["y"]), 1)
        exp_op = np.array([135.0, 170.0])
        self.assertEqual((op == exp_op).all(), True)

    @ignore_warnings
    def test_ts_endog(self):
        """GLMNet: Test for timeseries with endogenous variable"""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip = df_ip.set_index("ts")
        mod = GLMNet(df=df_ip, y_var="y", method="timeseries")
        mod.predict(n_interval=10)
        metrics = mod.model_summary
        self.assertGreaterEqual(metrics["rsq"], 0.7)
        self.assertLessEqual(metrics["mape"], 0.8)

    @ignore_warnings
    def test_ts_exog(self):
        """GLMNet: Test for timeseries with exogenous variable"""
        x_var = ["cost"]
        y_var = "y"
        test_perc = 0.2
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="exog")
        df_ip = df_ip.set_index("ts")
        df_train = df_ip.iloc[0:int(len(df_ip) * (1-test_perc)), :]
        df_test = df_ip.iloc[int(len(df_ip) * (1-test_perc)): len(df_ip), :]
        df_test = df_test[x_var]
        mod = GLMNet(df_train, y_var, x_var, method="timeseries")
        mod.predict(df_test)
        metrics = mod.model_summary
        self.assertAlmostEqual(metrics["rsq"], 1, places=1)
        self.assertLessEqual(metrics["mape"], 0.1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
