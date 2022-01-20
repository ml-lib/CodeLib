"""
utils.

Utlities module
-----------------

**Available functions:**

- :``elapsed_time``: Function to return elapsed time.

Credits
-------
::

    Authors:
        - Diptesh

    Date: May 22, 2020
"""

# =============================================================================
# --- Import libraries
# =============================================================================

import time

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

# pylint: disable=invalid-name
# pylint: disable=abstract-class-instantiated

# =============================================================================
# --- User defined functions
# =============================================================================


def elapsed_time(text: str,
                 start_t: int,
                 sept: int = 70
                 ) -> str:
    """
    Return elapsed time.

    Parameters
    ----------
    :text: str

        Text to be printed

    :start_t: int

        Generated from time.time_ns()

    :sept: int

        Length of text

    Returns
    -------
    str
        A string containing arg "text" followed by hours, minutes, seconds,
        milliseconds.

    Example usage
    -------------

    >>> import time
    >>> start = time.time_ns()
    >>> time.sleep(2)
    >>> elapsed_time("Time taken:", start)
    Time taken: 00:00:02 000 ms

    """
    second, ms = divmod(round((time.time_ns() / 1e6) - (start_t / 1e6), 0),
                        1000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    fn_op = text + str("%02d:%02d:%02d %03d ms" % (hour, minute, second, ms))\
        .rjust(sept - len(text))
    return fn_op
