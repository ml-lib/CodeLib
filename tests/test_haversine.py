"""
Test suite module for ``metrics``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Feb 14, 2024
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib import haversine  # noqa: F841

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


class TestHaversine(unittest.TestCase):
    """Test suite for module ``haversine``."""

    def setUp(self):
        """Set up for module ``haversine``."""

    def test_haversine_distance_miles(self):
        """Haversine: Test for distance (miles)"""
        lon1 = np.array([20.0])
        lat1 = np.array([10.0])
        lon2 = np.array([25.0])
        lat2 = np.array([5.0])
        exp_op_mile = 487.0
        op_mile = haversine.haversine_cy(lon1, lat1, lon2, lat2, dist="mi")
        self.assertEqual(np.round(op_mile[0]), exp_op_mile)

    def test_haversine_distance_km(self):
        """Haversine: Test for distance (km)"""
        lon1 = np.array([20.0])
        lat1 = np.array([10.0])
        lon2 = np.array([25.0])
        lat2 = np.array([5.0])
        exp_op_km = 783.0
        op_km = haversine.haversine_cy(lon1, lat1, lon2, lat2, dist="km")
        self.assertEqual(np.round(op_km[0]), exp_op_km)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
