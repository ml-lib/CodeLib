"""
Time series module.

**Available routines:**

- class ``AutoArima``: Builds time series model using SARIMAX.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 05, 2022
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=R0902,R0903,W0511

from inspect import getsourcefile
from os.path import abspath

from typing import Dict, List

import re
import sys

import pandas as pd
import numpy as np

import pmdarima as pm

from statsmodels.tsa.seasonal import seasonal_decompose

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

# =============================================================================
# ---
# =============================================================================


class AutoArima():
    """Auto ARIMA time series module.

    Parameters
    ----------
    df: pandas.DataFrame

        Pandas dataframe containing the `y_var` and optinal `x_var`. The index
        **must** be a datetime with no missing periods.

    y_var: str

        Dependant variable

    x_var: List[str], optional

        Independant variables (the default is None).

    param: dict, optional

        Time series parameters (the default is None). If no parameters are
        passed the following is set as parameters::

                  max_p: 15,
                  max_d: 2,
                  max_q: 15,
                  max_P: 15,
                  max_D: 2,
                  max_Q: 15,
                  seasonal: seasonal,
                  m: m,
                  threshold: 0.05,
                  debug: False

    Returns
    -------
    model: object

        Final optimal model.

    model_summary: Dict

        Model summary with optimal parameters.

    y_hat: list

        Predicted values for the orginal data.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = AutoArima(df=df_ip,
                        y_var="y",
                        x_var=["cost", "stock_level", "retail_price"])
    >>> df_op = mod.predict(x_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 param: Dict[str, object] = None
                 ):
        """Initialize variables."""
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        self.param = param
        self.y_hat = None
        # Set default parameters
        if self.param is None:
            self.param = self._seasonality()
        # Build optimal model
        self.model = self._opt_param()
        self.opt_param = self.model.to_dict()
        # Compute metrics
        self.model_summary = self._compute_metrics()

    def _seasonality(self) -> Dict[str, object]:
        """Determine seasonality and return parameters."""
        decomposition = seasonal_decompose(self.df[self.y_var],
                                           model="additive")
        _seasonal = decomposition.seasonal
        freq = _seasonal.value_counts()
        m = int(np.ceil(len(self.df) / freq.iloc[0]))
        seasonal = True
        if m < 2:  # pragma: no cover
            seasonal = False
        param = {"max_p": 15,
                 "max_d": 2,
                 "max_q": 15,
                 "max_P": 15,
                 "max_D": 2,
                 "max_Q": 15,
                 "seasonal": seasonal,
                 "m": m,
                 "threshold": 0.05,
                 "debug": False}
        return param

    def _opt_param(self) -> object:
        if self.x_var is None:
            model = pm.auto_arima(y=self.df[[self.y_var]],
                                  start_p=0,
                                  max_p=self.param["max_p"],
                                  max_d=self.param["max_d"],
                                  start_q=0,
                                  max_q=self.param["max_q"],
                                  start_P=0,
                                  max_P=self.param["max_P"],
                                  max_D=self.param["max_D"],
                                  start_Q=0,
                                  max_Q=self.param["max_Q"],
                                  information_criterion="aicc",
                                  alpha=self.param["threshold"],
                                  trace=self.param["debug"],
                                  seasonal=self.param["seasonal"],
                                  m=self.param["m"])
        else:
            model = pm.auto_arima(y=self.df[[self.y_var]],
                                  X=self.df[self.x_var],
                                  start_p=0,
                                  max_p=self.param["max_p"],
                                  max_d=self.param["max_d"],
                                  start_q=0,
                                  max_q=self.param["max_q"],
                                  start_P=0,
                                  max_P=self.param["max_P"],
                                  max_D=self.param["max_D"],
                                  start_Q=0,
                                  max_Q=self.param["max_Q"],
                                  information_criterion="aicc",
                                  alpha=self.param["threshold"],
                                  trace=self.param["debug"],
                                  seasonal=self.param["seasonal"],
                                  m=self.param["m"])
        return model

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute commonly used metrics to evaluate the model."""
        y = self.df[[self.y_var]].iloc[:, 0].values.tolist()
        if self.x_var is None:
            d = self.opt_param["order"][1]
            y_hat = list(self.model.predict_in_sample(start=d,
                                                      end=len(self.df)))
        else:
            exog = self.df[self.x_var]
            y_hat = list(self.model.predict(n_periods=len(exog), X=exog))
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.y_hat = y_hat
        return model_summary

    def predict(self,
                x_predict: pd.DataFrame = None,
                n_interval: int = 1) -> pd.DataFrame:
        """Predict module.

        Parameters
        ----------
        x_predict : pd.DataFrame, optional

            Pandas dataframe containing `x_var` (the default is None).

        n_interval : int, optional

            Number of time period to predict (the default is 1).

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing `y_var` and `x_var` (optional).

        """
        if self.x_var is None:
            df_pred = self.model.predict(n_periods=n_interval,
                                         alpha=self.param["threshold"],
                                         return_conf_int=False)
            df_pred = pd.DataFrame(df_pred)
            df_pred.columns = [self.y_var]
        else:
            n_interval = x_predict.shape[0]
            df_pred = self.model.predict(n_periods=n_interval,
                                         X=x_predict,
                                         alpha=self.param["threshold"],
                                         return_conf_int=False)
            df_pred = pd.DataFrame(df_pred)
            df_pred = pd.concat([df_pred, x_predict.reset_index(drop=True)],
                                axis=1,
                                ignore_index=True)
            df_pred.columns = [self.y_var] + self.x_var
        return df_pred
