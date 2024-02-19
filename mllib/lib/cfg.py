"""
cfg.

Configuration file
------------------

Create module level variables for module ``mllib/lib``.

Input
-----

Change the following::

    __version__ : str
    __doc__     : str
    module      : str

Output
------

The file sets the following variables:

>>> __version__
>>> __doc__
>>> module
>>> hdfs
>>> path

Credits
-------
::

    Authors:
        - Diptesh

    Date: Feb 11, 2024
"""

# pylint: disable=invalid-name

import socket
import re

from inspect import getsourcefile
from os.path import abspath

__version__: str = "0.5.0"
__doc__: str = "Machine Learning Library"
module: str = "mllib"

# Set environment
hdfs: bool = bool(re.match(r"[a-z0-9]+\..+\.com", socket.gethostname()))

# Set module path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/" + module + ".*)", r"\1/", path)
