from distutils.core import setup, Extension

setup(name='pyRF_prob',
 version='1.0',
  ext_modules=[Extension(name='pyRF_prob',
   sources=['cdf.c', 'prob.c'],
    depends=['prob.h'])])