#!/bin/bash
root="@ROOT@"
venv="$root/venv/phd"
[ -f "$venv/bin/activate" ] && [ -z "$VIRTUAL_ENV" ] && source "$venv/bin/activate"

export CC=clang
export CXX=clang++

export PYTHONPATH=$root:$root/lib:$root/bazel-genfiles

alias pgit="git -C ~/phd"

# Run all python _test files in a directory.
pytest_dir() {
  for f in $(find $1 -name '*_test.py'); do
    echo $f
    if ! python $f ; then
      echo "$f tests failed" >&2
      break
    fi
  done
}
