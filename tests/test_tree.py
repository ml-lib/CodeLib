"""
Test suite module for ``XGBoost``.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 27, 2021
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

from sklearn.model_selection import train_test_split as split
from sklearn import metrics as sk_metrics

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.tree import RandomForest  # noqa: F841
from mllib.lib.tree import XGBoost  # noqa: F841

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


class Test_RandomForest(unittest.TestCase):
    """Test suite for module ``RandomForest``."""

    def setUp(self):
        """Set up for module ``RandomForest``."""

    def test_rf_class(self):
        """RandomForest: Test for classification."""
        x_var = ["x1", "x2", "x3", "x4"]
        y_var = "y"
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[[y_var] + x_var]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip[y_var],
                                  test_size=0.2,
                                  random_state=42)
        mod = RandomForest(df_train, y_var, x_var, method="classify")
        y_hat = mod.predict(df_test[x_var])[y_var].tolist()
        y = df_test[y_var].values.tolist()
        acc = round(sk_metrics.accuracy_score(y, y_hat), 2)
        self.assertGreaterEqual(acc, 0.93)

    @ignore_warnings
    def test_rf_reg(self):
        """RandomForest: Test for regression."""
        x_var = ["x1", "x2", "x3", "x4"]
        y_var = "y"
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[[y_var] + x_var]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip[y_var],
                                  test_size=0.2,
                                  random_state=42)
        mod = RandomForest(df_train, y_var, x_var, method="regression")
        y_hat = mod.predict(df_test[x_var])[y_var].tolist()
        y = df_test[y_var].values.tolist()
        mse = round(sk_metrics.mean_squared_error(y, y_hat), 2)
        self.assertLessEqual(mse, 0.1)


class Test_XGBoost(unittest.TestCase):
    """Test suite for module ``XGBoost``."""

    def setUp(self):
        """Set up for module ``XGBoost``."""

    @ignore_warnings
    def test_xgboost_class(self):
        """XGBoost: Test for classification."""
        x_var = ["x1", "x2"]
        y_var = "y"
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[[y_var] + x_var]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip[y_var],
                                  test_size=0.2,
                                  random_state=1)
        mod = XGBoost(df_train, y_var, x_var, method="classify")
        y_hat = mod.predict(df_test[x_var])[y_var].tolist()
        y = df_test[y_var].values.tolist()
        acc = round(sk_metrics.accuracy_score(y, y_hat), 2)
        self.assertGreaterEqual(acc, 0.93)

    @ignore_warnings
    def test_xgboost_reg(self):
        """XGBoost: Test for regression."""
        x_var = ["x1", "x2", "x3", "x4"]
        y_var = "y"
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[[y_var] + x_var]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip[y_var],
                                  test_size=0.2,
                                  random_state=1)
        mod = XGBoost(df_train, y_var, x_var, method="regression")
        y_hat = mod.predict(df_test[x_var])[y_var].tolist()
        y = df_test[y_var].values.tolist()
        mse = round(sk_metrics.mean_squared_error(y, y_hat), 2)
        self.assertLessEqual(mse, 0.5)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
