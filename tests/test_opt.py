"""
Unit tests for opt.py module.

Author
------
::

    Author: Diptesh Basak
    Date: Jun 16, 2019
    License: BSD 3-Clause
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
import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.opt import TSP  # noqa: F841
from mllib.lib.opt import Transport  # noqa: F841

# =============================================================================
# --- User defined functions
# =============================================================================


class Test_TSP(unittest.TestCase):
    """Test suite for TSP module."""

    def setUp(self):
        """Setup for module ``Test_TSP``."""
        warnings.simplefilter('ignore')

    def test_ip(self):
        """TSP: Integer programming"""
        df_ip = pd.read_csv(path + "/data/input/us_city.csv")
        df_ip = df_ip.iloc[:10, :]
        tsp = TSP()
        opt = tsp.solve(loc=df_ip["city"].tolist(),
                        lat=df_ip["lat"].tolist(),
                        lon=df_ip["lng"].tolist(),
                        debug=False)
        self.assertEqual(np.round(opt[1], 0), 626.0)

    def test_nn(self):
        """TSP: Nearest neighbor algorithm"""
        df_ip = pd.read_csv(path + "/data/input/us_city.csv")
        df_ip = df_ip.iloc[:50, :]
        tsp = TSP()
        opt = tsp.solve(loc=df_ip["city"].tolist(),
                        lat=df_ip["lat"].tolist(),
                        lon=df_ip["lng"].tolist(),
                        debug=False)
        self.assertEqual(np.round(opt[1], 0), 1402.0)


class Test_TP(unittest.TestCase):
    """Test suite for transportation problem."""

    def setUp(self):
        """Setup for module ``Test_TP``."""
        warnings.simplefilter('ignore')

    def test_transport_balanced(self):
        """TP: Balanced problem"""
        c_loc = ["1", "5", "10", "11", "100", "127", "324"]
        c_demand = [20, 10, 15, 0, 0, 25, 0]
        c_supply = [0, 0, 0, 30, 12, 0, 28]
        c_lat = [42.1, 43.0, 40.3, 46.8, 43.9, 41.6, 45.2]
        c_lon = [-102.1, -103.0, -100.3, -106.8, -103.9, -101.6, -105.2]
        prob = Transport(c_loc, c_demand, c_supply, c_lat, c_lon, 1)
        opt_out = prob.solve(0)
        self.assertEqual(np.round(prob.output[1], decimals=2), 23856.39)
        exp_op = [('100', '1', 2),
                  ('100', '5', 10),
                  ('11', '10', 15),
                  ('11', '127', 15),
                  ('324', '1', 18),
                  ('324', '127', 10)]
        self.assertEqual(opt_out, exp_op)

    def test_transport_unbalanced_demand(self):
        """TP: Unbalanced problem when Demand > Supply"""
        c_loc = ["1", "5", "10", "11", "100", "127", "324"]
        c_demand = [20, 10, 15, 0, 0, 250, 0]
        c_supply = [0, 0, 0, 30, 12, 0, 28]
        c_lat = [42.1, 43.0, 40.3, 46.8, 43.9, 41.6, 45.2]
        c_lon = [-102.1, -103.0, -100.3, -106.8, -103.9, -101.6, -105.2]
        prob = Transport(c_loc, c_demand, c_supply, c_lat, c_lon, 1)
        opt_out = prob.solve(0)
        self.assertEqual(np.round(prob.output[1], decimals=2), 22170.26)
        exp_op = [('100', '1', 2),
                  ('100', '5', 10),
                  ('11', '127', 30),
                  ('324', '1', 18),
                  ('324', '127', 10),
                  ('Dummy', '10', 15),
                  ('Dummy', '127', 210)]
        self.assertEqual(opt_out, exp_op)

    def test_transport_unbalanced_supply(self):
        """TP: Unbalanced problem when Supply > Demand"""
        c_loc = ["1", "5", "10", "11", "100", "127", "324"]
        c_demand = [20, 10, 15, 0, 0, 25, 0]
        c_supply = [0, 0, 0, 30, 12, 0, 280]
        c_lat = [42.1, 43.0, 40.3, 46.8, 43.9, 41.6, 45.2]
        c_lon = [-102.1, -103.0, -100.3, -106.8, -103.9, -101.6, -105.2]
        prob = Transport(c_loc, c_demand, c_supply, c_lat, c_lon, 1)
        opt_out = prob.solve(0)
        self.assertEqual(np.round(prob.output[1], decimals=2), 19822.6)
        exp_op = [('100', '1', 2),
                  ('100', '5', 10),
                  ('11', 'Dummy', 30),
                  ('324', '1', 18),
                  ('324', '10', 15),
                  ('324', '127', 25),
                  ('324', 'Dummy', 222)]
        self.assertEqual(opt_out, exp_op)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
