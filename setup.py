from setuptools import setup

with open('./requirements.txt') as infile:
    requirements = [x.strip() for x in infile.readlines() if x.strip()]

setup(name="me.csv",
      version="0.0.1.dev1",
      description="Aggregate time and health tracking data",
      url="https://github.com/ChrisCummins/me.csv",
      author="Chris Cummins",
      author_email="chrisc.101@gmail.com",
      license="MIT License",
      packages=["me"],
      entry_points={'console_scripts': ['me.csv=me.cli:main']},
      test_suite="nose.collector",
      tests_require=["nose"],
      install_requires=requirements,
      zip_safe=True)