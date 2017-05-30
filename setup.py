from setuptools import setup

with open('./requirements.txt') as infile:
    requirements = [x.strip() for x in infile.readlines() if x.strip()]

setup(name='lmk',
      version='0.0.8',
      description='Email notification when command completes',
      url='https://github.com/ChrisCummins/lmk',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='GNU General Public License, Version 3',
      scripts=['lmk'],
      install_requires=requirements,
      zip_safe=True)
