from distutils.core import setup, Extension
setup(name='pyRF_prob', version='1.0', ext_modules=[Extension('pyRF_prob', ['cdf.c'])])