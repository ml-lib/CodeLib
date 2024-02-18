"""
Common metrics required in machine learning modules.

**Available functions:**
    - ``rsq``: R-Squared
    - ``mse``: Mean squared error
    - ``rmse``: Root mean squared error
    - ``mae``: Mean absolute error
    - ``mape``: Mean absolute percentage error
    - ``aic``: Akaike information criterion

Credits
-------
::

    Authors:
        - Diptesh

    Date: Dec 19, 2021
"""

from libc.math cimport log, sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()
DTYPE = np.double
ctypedef cnp.double_t DTYPE_t

# =============================================================================
# --- User defined functions
# =============================================================================

# =============================================================================
# R-Squared
# =============================================================================
cpdef double rsq(cnp.ndarray[DTYPE_t, ndim=1] y,
                 cnp.ndarray[DTYPE_t, ndim=1] y_hat):
    """
    Compute `Coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    or R-Squared.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    Returns
    -------
    op: float

        R-Squared value.

    """
    cdef double op = 0.0
    cdef Py_ssize_t arr_len = y.shape[0]
    cdef int i = 0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double y_sum = 0.0
    cdef double y_mean = 0.0
    cdef double num = 0.0
    cdef double den = 0.0
    for i in range(0, arr_len, 1):
        a = y[i]
        y_sum += a
    y_mean = y_sum * arr_len ** -1.0
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        num += (a - b) ** 2
        den += (a - y_mean) ** 2
    if den != 0.0:
        op = 1 - (num * den ** -1.0)
    return op

# =============================================================================
# Mean squared error
# =============================================================================
cpdef double mse(cnp.ndarray[DTYPE_t, ndim=1] y,
                 cnp.ndarray[DTYPE_t, ndim=1] y_hat):
    """
    Compute `Mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    Returns
    -------
    op: float

        Mean squared error.

    """
    cdef double op = 0.0
    cdef Py_ssize_t arr_len = y.shape[0]
    cdef int i
    cdef double a
    cdef double b
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op = op + (a - b) ** 2
    op = op * arr_len ** -1.0
    return op

# =============================================================================
# Root mean squared error
# =============================================================================
cpdef double rmse(cnp.ndarray[DTYPE_t, ndim=1] y,
                  cnp.ndarray[DTYPE_t, ndim=1] y_hat):
    """
    Compute `Root mean square error
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    Returns
    -------
    op: float

        Root mean square error.

    """
    return mse(y, y_hat) ** 0.5

# =============================================================================
# Mean absolute error
# =============================================================================
cpdef double mae(cnp.ndarray[DTYPE_t, ndim=1] y,
                 cnp.ndarray[DTYPE_t, ndim=1] y_hat):
    """
    Compute `Mean absolute error
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    Returns
    -------
    op: float

        Mean absolute error.

    """
    cdef double op = 0.0
    cdef Py_ssize_t arr_len = y.shape[0]
    cdef int i
    cdef double a
    cdef double b
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op += abs(a - b)
    op = op * arr_len ** -1.0
    return op

# =============================================================================
# Mean absolute percentage error
# =============================================================================
cpdef double mape(cnp.ndarray[DTYPE_t, ndim=1] y,
                  cnp.ndarray[DTYPE_t, ndim=1] y_hat):
    """
    Compute `Mean absolute percentage error
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    Returns
    -------
    op: float

        Mean absolute percentage error.

    """
    cdef double op = 0.0
    cdef Py_ssize_t arr_len = y.shape[0]
    cdef int i
    cdef double a
    cdef double b
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        if a != 0.0:
          op += abs(1 - (b * a ** -1.0))
    op = op * arr_len ** -1.0
    return op

# =============================================================================
# Akaike information criterion
# =============================================================================
cpdef double aic(cnp.ndarray[DTYPE_t, ndim=1] y,
                 cnp.ndarray[DTYPE_t, ndim=1] y_hat,
                 int k=1,
                 str method="linear"):
    """
    Compute `Akaike information criterion
    <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_.

    Parameters
    ----------
    y: numpy.ndarray

        Actual values.

    y_hat: numpy.ndarray

        Predicted values.

    k: int, optional

        Number of parameters (the default is 1).

    method: str, optional

        Type of regression (the default is linear).

    Returns
    -------
    op: float

        Akaike information criterion.

    """
    cdef double op = 0.0
    cdef Py_ssize_t arr_len = y.shape[0]
    cdef double sse = 0.0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double small_sample = 0.0
    small_sample = arr_len * k ** -1
    if method == "linear":
        for i in range(0, arr_len, 1):
            a = y[i]
            b = y_hat[i]
            sse += (a - b) ** 2
        op = 2 * k - 2 * log(sse)
        if small_sample <= 40:
            op += (2 * k * (k + 1)) * (arr_len - k - 1) ** -1
    return op
