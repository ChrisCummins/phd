from setuptools import setup
from sys import version_info

# Python weka wrapper currently only supports Python2.
python_weka_wrapper = "python-weka-wrapper" if version_info[0] == 2 else ""

setup(name='labm8',
      version='0.0.1',
      description='A collection of utilities for collecting and manipulating quantitative experimental data',
      url='https://github.com/ChrisCummins/labm8',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='GPL v3',
      packages=['labm8'],
      test_suite='nose.collector',
      tests_require=[
          'coverage',
          'nose'
      ],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          python_weka_wrapper
      ],
      zip_safe=False)
