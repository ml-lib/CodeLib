"""
GLMNet module.

**Available routines:**

- class ``GLMNet``: Builds GLMnet model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 28, 2022
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

from scipy.stats import norm
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit as ts_split
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


def ignore_warnings(test_func):
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

    method : str, optional

        Can be either `timeseries` or `regression`
        (the default is regression)

    strata : pd.DataFrame, optional

        A pandas dataframe column defining the strata (the default is None).

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    param : Dict, optional

        GLMNet parameters (the default is None).
        In case of None, the parameters will default to::

            seed: 1
            a_inc: 0.05
            n_jobs: -1

    ts_param : dict, optional

        GLMNet parameters for timeseries method (the default is None).
        In case of None, the parameters will default to::

            threshold: 0.05
            max_lag: 20
            ts_x_var: None
            ts_lag_var: None
            seasonality: None

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
    >>> mod = GLMNet(df=df_ip, y_var=["y"], x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 method: str = "regression",
                 strata: str = None,
                 k_fold: int = 5,
                 param: Dict = None,
                 ts_param: Dict = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df
        self.strata = strata
        self.method = method
        if self.method == "regression":
            self.df = self.df.reset_index(drop=True)
        self.k_fold = k_fold
        self.seed = 1
        if self.method == "timeseries":
            self.ts_param = ts_param
            if self.ts_param is None:
                self.ts_param = {}
                self.ts_param["threshold"] = 0.05
                self.ts_param["max_lag"] = 20
            self.ts_param["ts_x_var"] = None
            self.ts_param["ts_lag_var"] = None
            self._ts_data_transform()
            self.k_fold = ts_split(n_splits=self.k_fold)\
                .split(X=self.ts_df[self.y_var])
        self.model = None
        self.model_summary = None
        self.param = param
        if self.param is None:
            self.param = {"seed": 1,
                          "a_inc": 0.05,
                          "n_jobs": -1}
        self.param["l1_range"] = list(np.round(np.arange(self.param["a_inc"],
                                                         1.01,
                                                         self.param["a_inc"]),
                                               2))
        self._fit()
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        if self.method == "regression":
            y = self.df.loc[:, self.y_var].values.tolist()
            y_hat = list(self.model.predict(self.df[self.x_var]))
        elif self.method == "timeseries":
            y = self.ts_df.loc[:, self.y_var].values.tolist()
            y_hat = list(self.model.predict(
                self.ts_df[self.ts_param["ts_x_var"]]))
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    def _ts_data_transform(self):
        """Transform input data with significant lag variables."""
        # Determine seasonality and return seaonal lag
        decomposition = seasonal_decompose(self.df[self.y_var],
                                           model="additive")
        _seasonal = decomposition.seasonal
        freq = _seasonal.value_counts()
        self.ts_param["seasonality"] = \
            int(np.ceil(len(self.df) / freq.iloc[0]))
        # Determine significant lags
        df = self.df.copy(deep=True)
        df = df[self.y_var]
        df = pd.DataFrame({"lag": list(range(self.ts_param["max_lag"]+1)),
                           "pacf": pacf(df,
                                        nlags=self.ts_param["max_lag"],
                                        method='ols')})
        df["thres_val"] = \
            (np.round(norm.ppf(1 - (self.ts_param["threshold"] / 2)), 2)
             / (len(self.df) ** 0.5))
        df["pacf_sig"] = np.where((df['pacf'] >= df["thres_val"])
                                  | (df['pacf'] <= - df["thres_val"]),
                                  1, 0)
        df = df.where(df['pacf_sig'] > 0)
        df = df.dropna()
        self.ts_param["ts_lag_var"] = df['lag'].astype(int).to_list()
        self.ts_param["ts_lag_var"].append(self.ts_param["seasonality"])
        self.ts_param["ts_lag_var"] = \
            [x for x in self.ts_param["ts_lag_var"] if x != 0]
        self.ts_param["ts_lag_var"] = list(set(self.ts_param["ts_lag_var"]))
        self.ts_df = pd.DataFrame(self.df.loc[:, self.y_var])
        # TODO: Add integration test
        if len(self.ts_param["ts_lag_var"]) == 0:  # pragma: no cover
            self.ts_param["ts_lag_var"] = [1]
        for lag in self.ts_param["ts_lag_var"]:
            self.ts_df.loc[:, "lag_" + str(lag)] = \
                                    self.ts_df[self.y_var].shift(lag)
        if self.x_var is not None:
            self.ts_df = self.ts_df.join(self.df[self.x_var])
        self.ts_df = self.ts_df.dropna()
        self.ts_param["ts_x_var"] = list(self.ts_df.columns)
        self.ts_param["ts_x_var"].remove(self.y_var)

    # TODO: Remove this once GLMNet is updated
    @ignore_warnings
    def _fit(self) -> None:
        """Fit the best GLMNet model."""
        mod = ElasticNetCV(l1_ratio=self.param["l1_range"],
                           fit_intercept=True,
                           alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                   1.0, 10.0, 100.0],
                           normalize=True,
                           cv=self.k_fold,
                           n_jobs=self.param["n_jobs"],
                           random_state=self.param["seed"])
        if self.method == "timeseries":
            mod_op = mod.fit(self.ts_df[self.ts_param["ts_x_var"]],
                             self.ts_df[self.y_var].values.ravel())
        elif self.method == "regression":
            mod_op = mod.fit(self.df[self.x_var],
                             self.df[self.y_var].values.ravel())
        self.model = mod_op
        best_params_ = {"alpha": mod.l1_ratio_,
                        "lambda": mod.alpha_,
                        "intercept": mod.intercept_,
                        "coef": mod.coef_}
        self.model = mod
        self.best_params_ = best_params_

    def _ts_predict(self,
                    x_predict: pd.DataFrame = None,
                    n_interval: int = 1) -> pd.DataFrame:
        """Predict values for time series."""
        if self.x_var is None:
            df_op = [-1.0] * n_interval
            df_op = pd.DataFrame(df_op)
            df_op.columns = [self.y_var]
        else:
            df_op = x_predict.copy(deep=True)
            df_op[self.y_var] = -1.0
        lst_lag_val = self.df[self.y_var].tolist()
        for i, _ in enumerate(df_op.values):
            df_pred_x = pd.DataFrame(df_op.iloc[i]).T
            for j, _ in enumerate(self.ts_param["ts_lag_var"]):
                df_pred_x["lag_" + str(self.ts_param["ts_lag_var"][j])] \
                    = lst_lag_val[len(lst_lag_val)
                                  - self.ts_param["ts_lag_var"][j]]
            df_pred_x = pd.DataFrame(df_pred_x)
            y_hat = self.model.predict(df_pred_x[self.ts_param["ts_x_var"]])
            df_op.iloc[i, df_op.columns.get_loc(self.y_var)] = y_hat[0]
            lst_lag_val.append(y_hat[0])
        return df_op

    def predict(self,
                x_predict: pd.DataFrame = None,
                n_interval: int = 1) -> pd.DataFrame:
        """Predict values."""
        if self.method == "timeseries":
            df_op = self._ts_predict(x_predict, n_interval)
        else:
            df_op = x_predict.copy(deep=True)
            y_hat = self.model.predict(x_predict)
            df_op.insert(loc=0, column=self.y_var, value=y_hat)
        return df_op
