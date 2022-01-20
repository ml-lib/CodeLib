"""
Test suite module for ``model``.

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
import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.model import GLMNet  # noqa: F841

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
        op = mod.opt
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


# =============================================================================
# --- Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
