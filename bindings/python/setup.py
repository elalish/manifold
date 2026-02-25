# SPDX: unlicense
# Modified from https://github.com/maximiliank/cmake_python_r_example/blob/master/src/Python/setup.py.in

from setuptools import setup
from setuptools.dist import Distribution
import os
import sys
import sysconfig

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

package_files=['manifold3d.{sysconfig.get_config_var("EXT_SUFFIX")}']
if os.path.isfile('manifold3d.pyi'):
    package_files.append('manifold3d.pyi')

setup(
    include_package_data=True,
    package_data={
        '': package_files
    },
    distclass=BinaryDistribution
)
