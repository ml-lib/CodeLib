"""
Tree based models.

**Available routines:**

- class ``RandomForest``: Builds Random Forest model using cross validation.
- class ``XGBoost``: Builds XGBoost model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 15, 2022
"""

# pylint: disable=invalid-name
# pylint: disable=W0511,R0902,R0903,R0913,C0413

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np
import sklearn.ensemble as rf
import xgboost as xgb

from scipy.stats import norm
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit as ts_split
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


class Tree():
    """Parent class for tree based models."""

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 method: str = "regression",
                 k_fold: int = 5,
                 param: Dict = None,
                 ts_param: Dict = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df
        self.method = method
        if self.method in ("classify", "regression"):
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
        self.best_params_ = self._fit()
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        if self.method in ("classify", "regression"):
            y = self.df.loc[:, self.y_var].values.tolist()
            y_hat = list(self.model.predict(self.df[self.x_var]))
        elif self.method == "timeseries":
            y = self.ts_df.loc[:, self.y_var].values.tolist()
            y_hat = list(self.model.predict(
                self.ts_df[self.ts_param["ts_x_var"]]))
        if self.method in ("regression", "timeseries"):
            model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                             "mae": np.round(metrics.mae(y, y_hat), 3),
                             "mape": np.round(metrics.mape(y, y_hat), 3),
                             "rmse": np.round(metrics.rmse(y, y_hat), 3)}
            model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        if self.method == "classify":
            class_report = classification_report(y,
                                                 y_hat,
                                                 output_dict=True,
                                                 zero_division=0)
            model_summary = class_report["weighted avg"]
            model_summary["accuracy"] = class_report["accuracy"]
            model_summary = {key: round(model_summary[key], 3)
                             for key in model_summary}
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

    def _fit(self) -> Dict[str, Any]:  # pragma: no cover
        """Fit model."""
        return self.best_params_

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
        for i, _ in enumerate(df_op):
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


class RandomForest(Tree):
    """Random forest module.

    Objective:
        - Build
          `Random forest <https://en.wikipedia.org/wiki/Random_forest>`_
          model and determine optimal k

    Parameters
    ----------
    df : pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    y_var : str

        Dependant variable

    x_var : List[str]

        Independant variables

    method : str, optional

        Can be either `classify`, `timeseries` or `regression`
        (the default is regression)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    threshold : float, optional

         Threshold to identify significant lag values (the default is 0.05)

    param : dict, optional

        Random forest parameters (the default is None).
        In case of None, the parameters will default to::

            bootstrap: [True]
            max_depth: [1, len(x_var)]
            n_estimators: [1000]
            max_features: ["sqrt", "auto"]
            min_samples_leaf: [2, 5]

    ts_param : dict, optional

        Random forest parameters (the default is None).
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
        MAPE for regression and Accuracy, Precision, Recall, F1 score for
        classification.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = RandomForest(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(x_predict)

    """

    def _fit(self) -> Dict[str, Any]:
        """Fit RandomForest model."""
        if self.param is None:
            self.param = {"bootstrap": [True],
                          "n_estimators": [100]}
            if self.method == "classify":
                self.param["max_features"] = ["sqrt"]
                self.param["min_samples_leaf"] = [2]
                self.param["max_depth"] = list(range(1, len(self.x_var)))
            elif self.method == "regression":
                self.param["max_features"] \
                    = [int(np.ceil(len(self.x_var) / 3))]
                self.param["min_samples_leaf"] = [5]
                self.param["max_depth"] \
                    = list(range(1, len(self.x_var)))
            elif self.method == "timeseries":
                self.param["max_features"] \
                    = [int(np.ceil(len(self.ts_param["ts_x_var"]) / 3))]
                self.param["min_samples_leaf"] = [5]
                self.param["max_depth"] = \
                    list(range(1, len(self.ts_param["ts_x_var"])))
        if self.method == "classify":
            tmp_model = rf.RandomForestClassifier(oob_score=True,
                                                  random_state=self.seed)
        elif self.method in ("regression", "timeseries"):
            tmp_model = rf.RandomForestRegressor(oob_score=True,
                                                 random_state=self.seed)
        gs = RandomizedSearchCV(estimator=tmp_model,
                                param_distributions=self.param,
                                n_jobs=-1,
                                verbose=0,
                                refit=True,
                                n_iter=3,
                                return_train_score=True,
                                cv=self.k_fold)
        if self.method == "timeseries":
            gs_op = gs.fit(self.ts_df[self.ts_param["ts_x_var"]],
                           self.ts_df[self.y_var])
        elif self.method in ("regression", "classify"):
            gs_op = gs.fit(self.df[self.x_var],
                           self.df[self.y_var])
        self.model = gs_op
        return gs_op.best_params_


class XGBoost(Tree):
    """XGBoost module.

    Objective:
        - Build
          `XGBoost <https://en.wikipedia.org/wiki/XGBoost>`_
          model and determine optimal k

    Parameters
    ----------
    df : pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    y_var : str

        Dependant variable

    x_var : List[str]

        Independant variables

    method : str, optional

        Can be either `classify`, `timeseries` or `regression`
        (the default is regression)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    threshold : float, optional

         Threshold to identify significant lag values (the default is 0.05)

    param : dict, optional

        XGBoost parameters (the default is None).
        In case of None, the parameters will default to::

            n_estimators: [100]
            learning_rate: [0.01, 0.1, 0.2, 0.3]
            subsample: [0.5, 0.75, 1.0]
            colsample_bytree: [0.5, 1.0]
            min_child_weight: [0.5, 1.0, 3.0]
            max_depth: [int(len(self.x_var) * 0.8]
            objective: ["reg:squarederror", "binary:logistic"]

    ts_param : dict, optional

        Random forest time series parameters (the default is None).
        In case of None, the parameters will default to::

            threshold: 0.05
            max_lag: 20
            ts_x_var: None
            ts_lag_var: None
            seasonlity: None

    Returns
    -------
    model : object

        Final optimal model.

    best_params_ : Dict

        Best parameters amongst the given parameters.

    model_summary : Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE for regression and Accuracy, Precision, Recall, F1 score for
        classification.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = XGBoost(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(x_predict)

    """

    def _fit(self) -> Dict[str, Any]:
        """Fit XGBoost model."""
        if self.param is None:
            self.param = {"n_estimators": [100],
                          "learning_rate": [0.01, 0.1, 0.2, 0.3],
                          "subsample": [0.5, 0.75, 1.0],
                          "colsample_bytree": [0.5, 1.0],
                          "min_child_weight": [0.5, 1.0, 3.0]}
            if self.method == "timeseries":
                self.param["max_depth"] = \
                    [int(len(self.ts_param["ts_x_var"]) * 0.8)]
            elif self.method in ("regression", "classify"):
                self.param["max_depth"] = [int(len(self.x_var) * 0.8)]
            if self.method == "classify":
                self.param["objective"] = ["binary:logistic"]
            elif self.method in ("regression", "timeseries"):
                self.param["objective"] = ["reg:squarederror"]
        if self.method == "classify":
            tmp_model = xgb.XGBClassifier(n_jobs=1,
                                          verbosity=0,
                                          silent=True,
                                          random_state=self.seed,
                                          seed=self.seed)
        elif self.method in ("regression", "timeseries"):
            tmp_model = xgb.XGBRegressor(n_jobs=1,
                                         verbosity=0,
                                         silent=True,
                                         random_state=self.seed,
                                         seed=self.seed)
        gs = RandomizedSearchCV(estimator=tmp_model,
                                param_distributions=self.param,
                                n_jobs=-1,
                                verbose=0,
                                refit=True,
                                n_iter=10,
                                return_train_score=True,
                                cv=self.k_fold,
                                random_state=self.seed)
        if self.method == "timeseries":
            gs_op = gs.fit(self.ts_df[self.ts_param["ts_x_var"]],
                           self.ts_df[self.y_var])
        elif self.method in ("regression", "classify"):
            gs_op = gs.fit(self.df[self.x_var],
                           self.df[self.y_var])
        self.model = gs_op
        return gs_op.best_params_
