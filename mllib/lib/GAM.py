"""
GAM module.

**Available routines:**

    - class ``GAM``: Builds GAM model using grid search

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 31, 2022
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413,W0122,W0511,W0611

from typing import List

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np

from pygam import LinearGAM, s, l  # noqa: F841

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


class GAM():
    """GAM module.

    Objective:
        - Build GAM model using grid search

    Parameters
    ----------
    df : pd.DataFrame

        Pandas dataframe containing `y_var` and `x_var` variables.

    y_var : str

        Dependant variable.

    x_var : List[str]

        Independant variables.

    linear_var : List[str]

        List of variables to fit linear (the default is None).

    cubic_var : List[str]

        List of variables to fit cubic order (the default is None).

    splines : int

        Number of splines to fit (the default is 100).

    lams : List[float]

        List of lambds for grid search (the default is None).
        In case of None, the parameters will default to::

            lams = [0.6, 10, 100, 1000]

    Returns
    -------
    model : object

        Final optimal model.

    best_params_ : Dict

        Best parameters amongst the given parameters.

    model_summary : Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE

    Methods
    -------
    predict

    Example
    -------
    >>> mod = GAM(df=df_ip, y_var=["y"], x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 linear_var: List[str] = None,
                 cubic_var: List[str] = None,
                 splines: int = 100,
                 lams: List[float] = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df[self.x_var + [self.y_var]]
        self.linear_var = linear_var
        if self.linear_var is None:
            self.linear_var = []
        self.cubic_var = cubic_var
        if self.cubic_var is None:
            self.cubic_var = []
        self.splines = splines
        self.lams = lams
        if self.lams is None:
            self.lams = [0.6, 10, 100, 1000]
        self.model = None
        self.model_summary = None
        self._fit()
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.loc[:, self.y_var].values.tolist()
        y_hat = list(self.model.predict(self.df[self.x_var]))
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    def _model(self) -> str:
        """Generate model object."""
        model = "LinearGAM("
        for i, x in enumerate(self.x_var):
            if x in self.linear_var:
                model += "l(" + str(i) + ") + "
            elif x in self.cubic_var:
                model += "s(" + str(i) + \
                         ", n_splines=self.splines, spline_order=3) +"
            else:
                model += "s(" + str(i) + ", n_splines=self.splines) + "
        model = model[0:len(model)-2]
        model += ")"
        model = "self.model = " + model + \
                ".gridsearch(np.array(self.df[self.x_var]), \
                             np.array(self.df[self.y_var]), lam = self.lams)"
        return model

    def _fit(self) -> None:
        """Fit the GAM model."""
        model = self._model()
        exec(model)
        self.best_params_ = {'lam': self.model.lam}

    def predict(self,
                x_predict: pd.DataFrame = None) -> pd.DataFrame:
        """Predict values."""
        df_op = x_predict.copy(deep=True)
        y_hat = self.model.predict(np.array(x_predict))
        df_op.insert(loc=0, column=self.y_var, value=y_hat)
        return df_op
