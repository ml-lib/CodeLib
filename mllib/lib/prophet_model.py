"""
Time series module.

**Available routines:**
    - class ``FBP``: Builds time series model using fbprophet.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Jan 30, 2022
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413,R0205

from typing import List, Dict, Any

import re
import sys
import os

from inspect import getsourcefile
from os.path import abspath

import itertools
import logging
import pandas as pd
import numpy as np

# import pystan
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off
logging.getLogger("prophet").setLevel(logging.ERROR)

# __all__ = ["pystan", ]

os.environ['NUMEXPR_MAX_THREADS'] = '8'


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class FBP():
    """prophet module.

    Parameters
    ----------
    df: pandas.DataFrame

        Pandas dataframe containing the `y_var`, `ds` and `x_var`

    y_var: str

        Dependant variable

    x_var: List[str], optional

        Independant variables (the default is None).

    ds: str, optional

        Column name of the date variable (the default is None).

    hols_country: str, optional

        Country for which holiday events should be considered
        (the default is None).

    holidays: pandad.DataFrame, optional

        Pandas dataframe containing the `holiday`, `ds`, 'lower_window'
        and 'upper_window'

    param: dict, optional, Not implemented yet

        Time series parameters (the default is None).

    Returns
    -------
    model: object

        Final optimal model.

    model_summary: Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = TimeSeries(df=df_ip,
                         y_var="y",
                         x_var=["cost", "stock_level", "retail_price"],
                         ds="ds")
    >>> df_op = mod.predict(x_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 ds: str = "ds",
                 hols_country: str = None,
                 holidays: pd.DataFrame = None,
                 param: Dict = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.ds = ds
        self.original_ds = ds
        self.df = df.reset_index(drop=True)
        if param is None:
            param = {"interval_width": [0.95],
                     "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5, 1],
                     "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0]}
        self.model = None
        self.model_summary = None
        self.hols_country = hols_country
        self.holidays = holidays
        self.param = param
        self._pre_processing()
        self._fit()
        self._compute_metrics()
        if x_var is not None:
            self.betas = self._regressor_coefficients(self.model)

    def _pre_processing(self):
        """Pre-process data."""
        self.df[self.ds] = pd.to_datetime(self.df[self.ds])
        if self.x_var is None:
            self.df = self.df[[self.ds] + [self.y_var]]
        else:
            self.df = self.df[[self.ds] + [self.y_var] + self.x_var]
        coln = list(self.df.columns)
        self.df.columns = ["ds", "y"] + coln[2:]
        self.y_var = "y"
        self.ds = "ds"

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.loc[:, self.y_var].values.tolist()
        if self.x_var is None:
            y_hat = list(self.model.predict(self.df[[self.ds]])["yhat"])
        else:
            y_hat = list(self.model.predict(self.df[[self.ds]
                                                    + self.x_var])["yhat"])
        y = np.array(y, dtype=float)
        y_hat = np.array(y_hat, dtype=float)
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    @staticmethod
    def _regressor_index(m, name):
        """
        Given the name of a regressor, return its index in the `beta` matrix.

        Parameters
        ----------
        m: object

            Prophet model object, after fitting.

        name: str

            Name of the regressor, as passed into the `add_regressor` function.

        Returns
        -------
        int

            The column index of the regressor in the `beta` matrix.

        """
        op = np.extract(m.train_component_cols[name] == 1,
                        m.train_component_cols.index)[0]
        return op

    def _regressor_coefficients(self, m):  # pragma: no cover
        """
        Summarise the coefficients of the extra regressors used in the model.

        For additive regressors, the coefficient represents the incremental
        impact on `y` of a unit increase in the regressor. For multiplicative
        regressors, the incremental impact is equal to `trend(t)` multiplied
        by the coefficient.
        Coefficients are measured on the original scale of the training data.

        Parameters
        ----------
        m: object

            Prophet model object, after fitting.

        Returns
        -------
        pd.DataFrame

        containing::

            regressor: Name of the regressor
            regressor_mode: Additive/multiplicative effect on y
            center: The mean of the regressor if standardized else 0
            coef_lower: Lower bound for the coefficient
            coef: Expected value of the coefficient
            coef_upper: Upper bound for the coefficient

        coef_lower/upper are estimated from MCMC samples.
        It is only different to coef if mcmc_samples > 0.

        """
        assert len(m.extra_regressors) > 0, 'No extra regressors found.'
        coefs = []
        for regressor, params in m.extra_regressors.items():
            beta = m.params['beta'][:, self._regressor_index(m, regressor)]
            if params['mode'] == 'additive':
                coef = beta * m.y_scale / params['std']
            else:
                coef = beta / params['std']
            percentiles = [
                (1 - m.interval_width) / 2,
                1 - (1 - m.interval_width) / 2,
            ]
            coef_bounds = np.quantile(coef, q=percentiles)
            record = {
                'regressor': regressor,
                'regressor_mode': params['mode'],
                'center': params['mu'],
                'coef_lower': coef_bounds[0],
                'coef': np.mean(coef),
                'coef_upper': coef_bounds[1],
            }
            coefs.append(record)
        return pd.DataFrame(coefs)

    def _model(self, params: dict) -> object:
        """Generate model object."""
        logging.getLogger('prophet').setLevel(logging.ERROR)
        with suppress_stdout_stderr():
            model = Prophet(holidays=self.holidays, **params)
        if self.x_var is not None:
            for var in self.x_var:
                model.add_regressor(var)
        if self.hols_country is not None:
            model.add_country_holidays(country_name=self.hols_country)
        with suppress_stdout_stderr():
            model.fit(self.df)
        return model

    def _fit(self) -> Dict[str, Any]:
        """Fit model."""
        # Find best params
        # Generate all combinations of parameters
        all_params = [dict(zip(self.param.keys(), param_key)) for param_key in
                      itertools.product(*self.param.values())]
        rmses = []
        for params in all_params:
            model = self._model(params)
            with suppress_stdout_stderr():
                df_cv = cross_validation(model, horizon='60 days')
                df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        best_param = \
            tuning_results[tuning_results.rmse
                           == tuning_results["rmse"].min()].iloc[0].to_dict()
        del best_param['rmse']
        model = self._model(best_param)
        self.model = model
        self.best_params_ = best_param

    def predict(self,
                x_predict: pd.DataFrame = None,
                n_interval: int = 1) -> pd.DataFrame:
        """Predict module.

        Parameters
        ----------
        x_predict : pd.DataFrame, optional

            Pandas dataframe containing `ds` and `x_var` (the default is None).

        n_interval : int, optional

            Number of time period to predict (the default is 1).

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing `y_var`, `ds` and `x_var`.

        """
        if self.x_var is None:
            x_predict = self.model.make_future_dataframe(periods=n_interval)
            x_predict = x_predict.iloc[-n_interval:, :]
        else:
            x_predict[self.original_ds] = \
                pd.to_datetime(x_predict[self.original_ds])
            x_predict = x_predict[[self.original_ds] + self.x_var]
            x_predict.rename(columns={self.original_ds: self.ds},
                             inplace=True)
        df_op = x_predict.copy(deep=True)
        forecast = self.model.predict(x_predict)
        y_hat = forecast['yhat'].values.tolist()
        df_op.insert(loc=0, column=self.y_var, value=y_hat)
        df_op.rename(columns={self.ds: self.original_ds}, inplace=True)
        return df_op
