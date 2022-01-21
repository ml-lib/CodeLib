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

    Date: Sep 27, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np
import sklearn.ensemble as rf
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


class Tree():
    """Parent class for tree based models."""

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "regression",
                 k_fold: int = 5,
                 param: Dict = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df.reset_index(drop=True)
        self.method = method
        self.k_fold = k_fold
        self.seed = 1
        self.model = None
        self.model_summary = None
        self.param = param
        self.best_params_ = self._fit()
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.loc[:, self.y_var].values.tolist()
        y_hat = list(self.model.predict(self.df[self.x_var]))
        if self.method == "regression":
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

    def _fit(self) -> Dict[str, Any]:  # pragma: no cover
        """Fit model."""
        return self.best_params_

    def predict(self, x_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict values."""
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

        Can be either `classify` or `regression` (the default is regression)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    param : dict, optional

        Random forest parameters (the default is None).
        In case of None, the parameters will default to::

            bootstrap: [True]
            max_depth: [1, len(x_var)]
            n_estimators: [1000]
            max_features: ["sqrt", "auto"]
            min_samples_leaf: [2, 5]

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
                          "max_depth": list(range(1, len(self.x_var))),
                          "n_estimators": [100]}
            if self.method == "classify":
                self.param["max_features"] = ["sqrt"]
                self.param["min_samples_leaf"] = [2]
            elif self.method == "regression":
                self.param["max_features"] = [int(len(self.x_var) / 3)]
                self.param["min_samples_leaf"] = [5]
        if self.method == "classify":
            tmp_model = rf.RandomForestClassifier(oob_score=True,
                                                  random_state=self.seed)
        elif self.method == "regression":
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

        Can be either `classify` or `regression` (the default is regression)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    param : dict, optional

        XGBoost parameters (the default is None).
        In case of None, the parameters will default to::

            n_estimators: [100]
            learning_rate: [0.01, 0.1, 0.2, 0.3]
            subsample: [0.5, 0.75, 1.0]
            colsample_bytree: [0.5, 1.0]
            min_child_weight: [0.5, 1.0, 3.0]
            max_depth: [int(len(self.x_var) * 0.8)]
            objective: ["reg:squarederror", "binary:logistic"]

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
                          "min_child_weight": [0.5, 1.0, 3.0],
                          "max_depth": [int(len(self.x_var) * 0.8)]}
            if self.method == "classify":
                self.param["objective"] = ["binary:logistic"]
            elif self.method == "regression":
                self.param["objective"] = ["reg:squarederror"]
        if self.method == "classify":
            tmp_model = xgb.XGBClassifier(n_jobs=1,
                                          verbosity=0,
                                          silent=True,
                                          random_state=self.seed,
                                          seed=self.seed,
                                          use_label_encoder=False)
        elif self.method == "regression":
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
        gs_op = gs.fit(self.df[self.x_var],
                       self.df[self.y_var])
        self.model = gs_op
        return gs_op.best_params_
