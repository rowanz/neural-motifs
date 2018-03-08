from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="draw_rectangles_cython", ext_modules=cythonize('draw_rectangles.pyx'), include_dirs=[numpy.get_include()])