from setuptools import setup

setup(
  name="freefocus",
  packages=["freefocus"],
  include_package_data=True,
  install_requires=["flask",],
  setup_requires=["pytest-runner",],
  tests_require=["pytest",],
)
