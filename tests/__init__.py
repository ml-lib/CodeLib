"""
Initialization file for unit tests.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 01, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import re
import sys

from inspect import getsourcefile
from os.path import abspath

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)
