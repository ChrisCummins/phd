from setuptools import setup

deps = [
  'PyGithub',
  'requests'
]

setup(name='smith',
      version='0.0.1',
      description='',
      url='https://github.com/ChrisCummins/phd',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='',
      packages=[ 'smith' ],
      package_data={ 'smith': [ 'share/sql/*.sql' ] },
      scripts=[
          'bin/fetch-clsmith',
          'bin/fetch-dnn',
          'bin/fetch-fs',
          'bin/fetch-gh',
          'bin/smith',
          'bin/smith-create-db',
          'bin/smith-explore',
          'bin/smith-explore-gh',
          'bin/smith-parboil',
          'bin/smith-preprocess',
          'bin/smith-train'
      ],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      install_requires=deps,
      data_files=[],
      zip_safe=False)
