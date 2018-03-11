#!/bin/bash
root="@ROOT@"
venv="$root/venv/phd"
[ -f "$venv/bin/activate" ] && [ -z "$VIRTUAL_ENV" ] && source "$venv/bin/activate"

export CC=clang
export CXX=clang++

export PYTHONPATH=$root:$root/lib:$root/bazel-genfiles

alias pgit="git -C ~/phd"
