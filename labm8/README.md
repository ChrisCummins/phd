![labm8](https://raw.github.com/ChrisCummins/labm8/master/labm8/labm8.jpg)

<!-- downloads counter -->
<a href="https://github.com/ChrisCummins/labm8">
  <img src="https://pepy.tech/badge/labm8">
</a>
<!-- pypi version -->
<a href="https://badge.fury.io/py/labm8">
    <img src="https://img.shields.io/pypi/v/labm8.svg?color=brightgreen">
</a>
<!-- Better code -->
<a href="https://bettercodehub.com/results/ChrisCummins/labm8">
  <img src="https://bettercodehub.com/edge/badge/ChrisCummins/labm8?branch=master">
</a>
<!-- Travis CI -->
<a href="https://github.com/ChrisCummins/labm8">
  <img src="https://img.shields.io/travis/ChrisCummins/labm8/master.svg">
</a>
<!-- commit counter -->
<a href="https://github.com/ChrisCummins/labm8">
  <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/labm8.svg?color=yellow">
</a>
<!-- repo size -->
<a href="https://github.com/ChrisCummins/labm8">
    <img src="https://img.shields.io/github/repo-size/ChrisCummins/labm8.svg">
</a>
<!-- license -->
<a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=brightgreen">
</a>

```sh
$ python -m pip install -U labm8
```

Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the Apache 2.0 license. See
`LICENSE` for details.


## Deployment Instructions

1. Set the desired version number in `//:version.txt`.

2. Export the source tree to the public [github repository](https://github.com/ChrisCummins/labm8):

```sh
$ bazel run //labm8/py:export
```

3. Deploy a new versioned release of the python package to [pypi](https://pypi.org/project/labm8/):

```sh
$ bazel run //labm8/py:export_python_pip -- --release_type=release
```
