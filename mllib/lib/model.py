"""
GLMNet module.

**Available routines:**

- class ``GLMNet``: Builds GLMnet model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 06, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413,W0511

from typing import List, Dict

import warnings
import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split as split

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


def ignore_warnings(test_func):  # pragma: no cover
    """Suppress warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class GLMNet():
    """GLMNet module.

    Objective:
        - Build
          `GLMNet <https://web.stanford.edu/~hastie/Papers/glmnet.pdf>`_
          model using optimal alpha and lambda

    Parameters
    ----------
    df : pd.DataFrame

        Pandas dataframe containing `y_var` and `x_var` variables.

    y_var : str

        Dependant variable.

    x_var : List[str]

        Independant variables.

    strata : pd.DataFrame, optional

        A pandas dataframe column defining the strata (the default is None).

    param : Dict, optional

        GLMNet parameters (the default is None).
        In case of None, the parameters will default to::

            seed: 1
            a_inc: 0.05
            test_perc: 0.25
            n_jobs: -1
            k_fold: 10

    Returns
    -------
    opt : Dict

        Summary of the model built along with best paramameters
        and estimators.

    model : object

        Final optimal model.

    model_summary : Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE

    Methods
    -------
    predict

    Example
    -------
    >>> mod = GLMNet(df=df_ip, y_var=["y"], x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 strata: str = None,
                 param: Dict = None):
        """Initialize variables for module ``GLMNet``."""
        self.df = df[[y_var] + x_var]
        self.y_var = y_var
        self.x_var = x_var
        self.strata = strata
        self.model_summary = None
        self.opt = None
        if param is None:
            param = {"seed": 1,
                     "a_inc": 0.05,
                     "test_perc": 0.25,
                     "n_jobs": -1,
                     "k_fold": 10}
        self.param = param
        self.param["l1_range"] = list(np.round(np.arange(self.param["a_inc"],
                                                         1.01,
                                                         self.param["a_inc"]),
                                               2))
        self._fit()
        self._compute_metrics()

    def _fit(self) -> None:
        """Fit the best GLMNet model."""
        train_x, test_x, \
            train_y, test_y = split(self.df[self.x_var],
                                    self.df[[self.y_var]],
                                    test_size=self.param["test_perc"],
                                    random_state=self.param["seed"],
                                    stratify=self.strata)
        mod = ElasticNetCV(l1_ratio=self.param["l1_range"],
                           fit_intercept=True,
                           alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                   1.0, 10.0, 100.0],
                           cv=self.param["k_fold"],
                           n_jobs=self.param["n_jobs"],
                           random_state=self.param["seed"])
        mod.fit(train_x, train_y.values.ravel())
        opt = {"alpha": mod.l1_ratio_,
               "lambda": mod.alpha_,
               "intercept": mod.intercept_,
               "coef": mod.coef_,
               "train_v": mod.score(train_x, train_y),
               "test_v": mod.score(test_x, test_y)}
        self.model = mod
        self.opt = opt

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df[[self.y_var]].iloc[:, 0].values.tolist()
        y_hat = list(self.predict(self.df[self.x_var])["y"].values)
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict y_var/target variable.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`.

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var` and `x_var`.

        """
        y_hat = self.model.predict(df_predict)
        df_predict.insert(loc=0, column=self.y_var, value=y_hat)
        return df_predict
