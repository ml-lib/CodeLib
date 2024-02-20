"""
Machine Learning Library.

Objective:
    - Illustrate module APIs with some examples.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 01, 2021
"""

# pylint: disable=invalid-name, E0611

# =============================================================================
# --- Import libraries
# =============================================================================

import argparse
import time

import pandas as pd

from lib import cfg, utils  # noqa: F841
from lib.cluster import Cluster  # noqa: F841
from lib.glmnet import GLMNet  # noqa: F841
from lib.knn import KNN  # noqa: F841
from lib.tree import RandomForest  # noqa: F841
from lib.tree import XGBoost  # noqa: F841
from lib.opt import TSP  # noqa: F841
from lib.opt import Transport  # noqa: F841
from lib.timeseries import AutoArima  # noqa: F841
from lib.timeseries import BatesGrager  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

__version__ = cfg.__version__
__doc__ = cfg.__doc__
path = cfg.path + "data/"
elapsed_time = utils.elapsed_time

sep = "-" * 70
print(sep, "\n" + __doc__, "v" + __version__, "\n" + sep + "\n")

# =============================================================================
# --- Arguments
#
# filename: str
# =============================================================================

CLI = argparse.ArgumentParser()

CLI.add_argument("-f", "--filename",
                 nargs=1,
                 type=str,
                 default=["iris.csv"],
                 help="input csv filename")

args = CLI.parse_args()

fn_ip = args.filename[0]

# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    start = time.time_ns()
    # --- KNN
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/iris.csv")
    mod = KNN(df_ip, "y", ["x1", "x2", "x3", "x4"], method="classify")
    print("KNN\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Clustering
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/iris.csv")
    clus_sol = Cluster(df=df_ip, x_var=["x1"])
    clus_sol.opt_k()
    print("\nClustering\n",
          "optimal k = " + str(clus_sol.optimal_k),
          elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- GLMNet
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/test_glmnet.csv")
    glm_mod = GLMNet(df=df_ip,
                     y_var="y",
                     x_var=["x1", "x2"])
    print("\nGLMNet\n")
    for k, v in glm_mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Random forest
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/iris.csv")
    x_var = ["x1", "x2", "x3", "x4"]
    y_var = "y"
    mod = RandomForest(df_ip, y_var, x_var, method="classify")
    print("\nRandom forest\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- XGBoost
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/iris.csv")
    x_var = ["x1", "x2", "x3", "x4"]
    y_var = "y"
    mod = XGBoost(df_ip, y_var, x_var, method="classify")
    print("\nXGBoost\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Travelling salesman
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/us_city.csv")
    df_ip = df_ip.iloc[:10, :]
    tsp = TSP()
    opt = tsp.solve(loc=df_ip["city"].tolist(),
                    lat=df_ip["lat"].tolist(),
                    lon=df_ip["lng"].tolist(),
                    debug=False)
    print("\nTravelling salesman problem\n")
    print("Optimal value:", round(opt[1], 3))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Transportation
    start_t = time.time_ns()
    c_loc = ["1", "5", "10", "11", "100", "127", "324"]
    c_demand = [20, 10, 15, 0, 0, 25, 0]
    c_supply = [0, 0, 0, 30, 12, 0, 28]
    c_lat = [42.1, 43.0, 40.3, 46.8, 43.9, 41.6, 45.2]
    c_lon = [-102.1, -103.0, -100.3, -106.8, -103.9, -101.6, -105.2]
    prob = Transport(c_loc, c_demand, c_supply, c_lat, c_lon, 1)
    opt_out = prob.solve(0)
    print("\nTransportation problem\n")
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Time series
    df_ip = pd.read_excel(path + "input/test_time_series.xlsx",
                          sheet_name="exog")
    df_ip = df_ip.set_index("ts")
    test_len = 30
    df_train = df_ip.iloc[:-test_len]
    df_test = df_ip.iloc[-test_len:]
    y_var = "y"
    x_var = ["cp", "stock_level", "retail_price"]
    df_op = pd.DataFrame()
    y = df_test[y_var].values.tolist()
    df_op["y"] = y
    op = {}
    # ---- ARIMA
    start_t = time.time_ns()
    mod = AutoArima(df=df_train, y_var=y_var, x_var=x_var)
    if x_var is not None:
        y_hat = mod.predict(df_test[x_var])[y_var].values.tolist()
    else:
        y_hat = mod.predict(n_interval=test_len)[y_var].values.tolist()
    df_op["arima"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["arima"] = tmp_op
    print("\nAutoArima timeseries\n")
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Random forest time series
    start_t = time.time_ns()
    mod = RandomForest(df_train, y_var=y_var, x_var=x_var, method="timeseries")
    print("\nRandom forest timeseries\n")
    if x_var is not None:
        y_hat = mod.predict(df_test[x_var])[y_var].values.tolist()
    else:
        y_hat = mod.predict(n_interval=test_len)[y_var].values.tolist()
    df_op["rf"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["rf"] = tmp_op
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- XGBoost time series
    start_t = time.time_ns()
    mod = XGBoost(df_train, y_var=y_var, x_var=x_var, method="timeseries")
    print("\nXGBoost timeseries\n")
    if x_var is not None:
        y_hat = mod.predict(df_test[x_var])[y_var].values.tolist()
    else:
        y_hat = mod.predict(n_interval=test_len)[y_var].values.tolist()
    df_op["xgb"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["xgb"] = tmp_op
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- GLMnet time series
    start_t = time.time_ns()
    mod = GLMNet(df_train, y_var=y_var, x_var=x_var, method="timeseries")
    print("\nGLMNet timeseries\n")
    if x_var is not None:
        y_hat = mod.predict(df_test[x_var])[y_var].values.tolist()
    else:
        y_hat = mod.predict(n_interval=test_len)[y_var].values.tolist()
    df_op["glmnet"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["glmnet"] = tmp_op
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Prophet
    start_t = time.time_ns()
    df_train = df_train.reset_index(drop=False)
    df_test = df_test.reset_index(drop=False)
    param = {"interval_width": [0.95],
             "changepoint_prior_scale": [0.1, 0.5],
             "seasonality_prior_scale": [0.01]}
    mod = FBP(df_train, y_var=y_var, x_var=x_var, ds="ts", param=param)
    print("\nProphet\n")
    if x_var is not None:
        y_hat = mod.predict(df_test[x_var + ['ts']])[y_var].values.tolist()
    else:
        y_hat = mod.predict(n_interval=test_len)[y_var].values.tolist()
    df_op["prophet"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["prophet"] = tmp_op
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- GAM
    start_t = time.time_ns()
    linear_var = x_var
    mod = GAM(df_train, y_var, x_var, linear_var=linear_var)
    print("\nGeneral additive models\n")
    y_hat = mod.predict(df_test[x_var])[y_var].values.tolist()
    df_op["gam"] = y_hat
    tmp_op = {"rsq": round(metrics.rsq(y, y_hat), 2),
              "mae": round(metrics.mae(y, y_hat), 2),
              "mape": round(metrics.mape(y, y_hat), 2),
              "mse": round(metrics.mse(y, y_hat), 2),
              "rmse": round(metrics.rmse(y, y_hat), 2)}
    op["gam"] = tmp_op
    for k, v in tmp_op.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Random forest time series
    start_t = time.time_ns()
    df_ip = pd.read_excel(path + "input/test_time_series.xlsx",
                          sheet_name="exog")
    df_ip = df_ip.set_index("ts")
    mod = RandomForest(df_ip, y_var="y", x_var=["cost"], method="timeseries")
    print("\nRandom forest timeseries\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- XGBoost time series
    start_t = time.time_ns()
    df_ip = pd.read_excel(path + "input/test_time_series.xlsx",
                          sheet_name="exog")
    df_ip = df_ip.set_index("ts")
    mod = XGBoost(df=df_ip, y_var="y", x_var=["cost"], method="timeseries")
    print("\nXGBoost timeseries\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- Bates & Granger
    start_t = time.time_ns()
    df_raw = pd.read_excel(path + "input/test_time_series.xlsx",
                           sheet_name="bates_granger")
    exp_op = df_raw[["ts", "y", "y_hat_01", "y_hat_02",
                     "y_hat_03", "y_hat_04", "y_hat_bg"]].fillna(0)
    df_ip = exp_op.drop("y_hat_bg", axis=1)
    mod = BatesGrager(df=df_ip,
                      y="y",
                      y_hat=["y_hat_01", "y_hat_02",
                             "y_hat_03", "y_hat_04"],
                      lag=53, pred_period=1)
    op = mod.solve()
    print("\nBates & Granger\n")
    for k, v in mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep,
          sep="\n")
    # --- EOF
    print(sep, elapsed_time("Total time", start), sep, sep="\n")
