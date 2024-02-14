"""
Test suite module for ``metrics``.

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

import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib import metrics  # noqa: F841

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


class TestMetrics(unittest.TestCase):
    """Test suite for module ``metrics``."""

    def setUp(self):
        """Set up for module ``metrics``."""

    def test_rsq(self):
        """Metrics: Test for R-squared"""
        y = [3, 8, 10, 17, 24, 27]
        y_hat = [2, 8, 10, 13, 18, 20]
        exp_op = 0.772
        op = np.round(metrics.rsq(y, y_hat), 3)
        self.assertEqual(op, exp_op)

    def test_mse(self):
        """Metrics: Test for MSE"""
        y = [34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]
        y_hat = [37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]
        exp_op = 5.917
        op = np.round(metrics.mse(y, y_hat), 3)
        self.assertEqual(op, exp_op)

    def test_rmse(self):
        """Metrics: Test for RMSE"""
        y = [34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]
        y_hat = [37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]
        exp_op = 2.432
        op = np.round(metrics.rmse(y, y_hat), 3)
        self.assertEqual(op, exp_op)

    def test_mae(self):
        """Metrics: Test for MAE"""
        y = [12, 13, 14, 15, 15, 22, 27]
        y_hat = [11, 13, 14, 14, 15, 16, 18]
        exp_op = 2.429
        op = np.round(metrics.mae(y, y_hat), 3)
        self.assertEqual(op, exp_op)

    def test_mape(self):
        """Metrics: Test for MAPE"""
        y = [34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]
        y_hat = [37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]
        exp_op = 0.065
        op = np.round(metrics.mape(y, y_hat), 3)
        self.assertEqual(op, exp_op)

    def test_aic_linear(self):
        """Metrics: Test for AIC in linear regression"""
        y = [34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]
        y_hat = [37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]
        exp_op = -6.125
        op = np.round(metrics.aic(y, y_hat, k=1, method="linear"), 3)
        self.assertEqual(op, exp_op)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
