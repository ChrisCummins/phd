from setuptools import setup

setup(name='labm8',
      version='0.0.1',
      description='A collection of utilities for collecting and manipulating quantitative experimental data',
      url='',
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
          'scipy'
      ],
      zip_safe=False)
