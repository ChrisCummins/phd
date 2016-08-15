from setuptools import setup

deps = [
    'editdistance',
    'numpy',
    'PyGithub',
    'pyopencl',
    'requests',
    'scikit-learn',
    'seaborn',
]

setup(name='smith',
      version='0.0.1',
      description='',
      url='https://github.com/ChrisCummins/phd',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='',
      packages=[ 'smith' ],
      package_data={ 'smith': [
          'share/include/*.h',
          'share/sql/*.sql'
      ] },
      scripts=[
          'bin/cecl2features',
          'bin/cgo13',
          'bin/fetch-clsmith',
          'bin/fetch-db',
          'bin/fetch-dnn',
          'bin/fetch-fs',
          'bin/fetch-gh',
          'bin/smith',
          'bin/smith-create-db',
          'bin/cldrive',
          'bin/smith-explore',
          'bin/smith-features',
          'bin/smith-parboil',
          'bin/smith-preprocess',
          'bin/smith-train'
      ],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      install_requires=deps,
      data_files=[],
      zip_safe=False)
