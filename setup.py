from setuptools import setup

setup(
    name='shutterbug',
    version='0.0.2',
    description='',
    url='https://github.com/ChrisCummins/shutterbug',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='GNU General Public License, Version 3',
    packages=['shutterbug'],
    scripts=['bin/shutterbug'],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[],
    data_files=[],
    zip_safe=True)
