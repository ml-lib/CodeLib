"""
Haversine distance formula.

**Available functions:**
    - ``haversine_cy``: Haversine distance

Credits
-------
::

    Authors:
        - Diptesh

    Date: Feb 14, 2024
"""

from libc.math cimport sin, cos, asin, sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()
DTYPE = np.double
ctypedef cnp.double_t DTYPE_t

cdef DTYPE_t deg2rad(DTYPE_t deg):
    """Convert degrees to radians."""
    cdef DTYPE_t rad = deg * 0.017453293
    return rad
    
cdef DTYPE_t c_haversine(DTYPE_t lon1,
                         DTYPE_t lat1,
                         DTYPE_t lon2,
                         DTYPE_t lat2,
                         str dist="mi"):
    """Compute Haversine distance for a pair of Lat/Lon."""
    cdef DTYPE_t rlon1 = deg2rad(lon1)
    cdef DTYPE_t rlon2 = deg2rad(lon2)
    cdef DTYPE_t rlat1 = deg2rad(lat1)
    cdef DTYPE_t rlat2 = deg2rad(lat2)
    
    cdef DTYPE_t dlon = rlon2 - rlon1
    cdef DTYPE_t dlat = rlat2 - rlat1

    cdef DTYPE_t a = sin(dlat/2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon/2)**2

    cdef DTYPE_t c = 2 * asin(sqrt(a))
    cdef DTYPE_t R = 0.0
    if dist == "km":  # pragma: no cover
        R = 6372.8
    else:
        R = 3959.87433
    cdef DTYPE_t tmp_op = R * c
    return tmp_op

cpdef cnp.ndarray[DTYPE_t,
                  ndim=1] haversine_cy(cnp.ndarray[DTYPE_t,
                                                   ndim=1] lon1,
                                       cnp.ndarray[DTYPE_t,
                                                   ndim=1] lat1,
                                       cnp.ndarray[DTYPE_t,
                                                   ndim=1] lon2,
                                       cnp.ndarray[DTYPE_t,
                                                   ndim=1] lat2,
                                       str dist="mi"):
    """
    Haversine distance formula.

    Calculate the euclidean distance in miles between two points
    specified in decimal degrees using
    `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float

        Pair of Latitude and Longitude. All args must be of equal length.

    dist : str, `optional`

        Output distance in miles ('mi') or kilometers ('km')
        (the default is mi)

    Returns
    -------
    numpy.ndarray

        Euclidean distance between two points in miles.

    """
    cdef Py_ssize_t arr_len = lon1.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] op = np.zeros([arr_len], dtype=DTYPE)
    cdef DTYPE_t a = 0.0
    cdef DTYPE_t b = 0.0
    cdef DTYPE_t c = 0.0
    cdef DTYPE_t d = 0.0
    cdef DTYPE_t e = 0.0
    for i in range(0, arr_len, 1):
        a = lon1[i]
        b = lat1[i]
        c = lon2[i]
        d = lat2[i]
        e = c_haversine(a, b, c, d, dist=dist)
        op[i] = e
    return op
