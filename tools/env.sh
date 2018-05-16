#!/bin/bash

root="@ROOT@"

# Export a dummy virtualenv.
# TODO(cec): Add a better way of signalling that we're in the phd env from
# the command line.
export VIRTUAL_ENV=phd

export CC=clang
export CXX=clang++

export PYTHONPATH=$root:$root/lib:$root/bazel-genfiles

alias pgit="git -C ~/phd"

alias dpack="python $root/lib/dpack/dpack.py"
