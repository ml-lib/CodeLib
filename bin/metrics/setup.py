"""Setup file."""

from setuptools import setup
from Cython.Build import cythonize

setup(name="Shared objects",
      ext_modules=cythonize("*.pyx"))
