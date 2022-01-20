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

import numpy as _np

from libc.math cimport log

# =============================================================================
# --- User defined functions
# =============================================================================


cpdef rsq(list y, list y_hat):
    """
    Compute `Coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    or R-Squared.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        R-Squared value.

    """
    cdef int i = 0
    cdef int arr_len = 0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double y_sum = 0.0
    cdef double y_mean = 0.0
    cdef double num = 0.0
    cdef double den = 0.0
    cdef double op = 0.0
    arr_len = len(y)
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


cpdef mse(list y, list y_hat):
    """
    Compute `Mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Parameters
    ----------
    :y: list

        Actual values.

    :y_hat: list

        Predicted values.

    Returns
    -------
    :op: float

        Mean squared error.

    """
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op = op + (a - b) ** 2
    op = op * arr_len ** -1.0
    return op

cpdef rmse(list y, list y_hat):
    """
    Compute `Root mean square error
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Root mean square error.

    """
    return mse(y, y_hat) ** 0.5


cpdef mae(list y, list y_hat):
    """
    Compute `Mean absolute error
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Mean absolute error.

    """
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op += abs(a - b)
    op = op * arr_len ** -1.0
    return op


cpdef mape(list y, list y_hat):
    """
    Compute `Mean absolute percentage error
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Mean absolute percentage error.

    """
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        if a != 0.0:
          op += abs(1 - (b * a ** -1.0))
    op = op * arr_len ** -1.0
    return op


cpdef double aic(list y, list y_hat, int k, str method="linear"):
    """
    Compute `Akaike information criterion
    <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    k : int

        Number of parameters.

    method : str, optional

        Type of regression (the default is linear).

    Returns
    -------
    op : float

        Akaike information criterion.

    """
    cdef double op = 0.0
    cdef double sse = 0.0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef int arr_len = 0
    cdef double small_sample = 0.0
    small_sample = arr_len * k ** -1
    arr_len = len(y)
    if method == "linear":
        for i in range(0, arr_len, 1):
            a = y[i]
            b = y_hat[i]
            sse += (a - b) ** 2
        op = 2 * k - 2 * log(sse)
        if small_sample <= 40:
            op += (2 * k * (k + 1)) * (arr_len - k - 1) ** -1
    return op
