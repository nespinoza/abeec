import re
import numpy
#from distutils.core import setup, Extension
from setuptools import setup, Extension

VERSIONFILE='src/_version.py'
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='abeec',
      version=verstr,
      description='abeec: a library to perform approximate bayesian computation',
      url='http://github.com/nespinoza/abec',
      author='Nestor Espinoza',
      author_email='nespinoza@stsci.edu',
      license='MIT',
      packages=['abeec'],
      package_dir={'abeec': 'src'},
      install_requires=['numpy','scipy'],
      python_requires='>=3.0',
      zip_safe=False)
