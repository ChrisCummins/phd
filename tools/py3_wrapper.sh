#!/bin/bash
# Select python 2 for bazel tools and python 3 for everything else.
#
# This is to workaround a known bug in the rules_docker bazel package, which
# depends on code which is not Python 3 compatible.
# See: https://github.com/bazelbuild/rules_docker/issues/293

if [[ $1 = *"/bazel_tools/"* ]] ||
  [[ $1 = *"/containerregistry/"* ]] ||
  [[ $1 = *"/external/puller/file/"* ]] ||
  [[ $1 = *"/io_bazel_rules_docker/"* ]]; then
  PYTHON_BIN=python2
else
  PYTHON_BIN=python3.6
fi

${PYTHON_BIN} "$@"
