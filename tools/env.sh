#!/bin/bash

export PHD="__ROOT__"

# Export a dummy virtualenv.
# TODO(cec): Add a better way of signalling that we're in the phd env from
# the command line.
export VIRTUAL_ENV=phd

export CC=clang
export CXX=clang++

export PYTHONPATH=$PHD:$PHD/lib:$PHD/bazel-genfiles

alias pgit="git -C ~/phd"

alias dpack="python $PHD/lib/dpack/dpack.py"
