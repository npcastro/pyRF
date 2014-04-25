from distutils.core import setup, Extension

setup(name='pyRF_prob',
 version='1.0',
  ext_modules=[Extension(name='pyRF_prob',
   sources=['Libraries/cdf.c', 'Libraries/prob.c'],
    depends=['Libraries/prob.h'])])

