#!/usr/bin/env bash
#
# Script to run a Jupyter notebook server in this directory.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
set -eu

# Find the root of the source tree by recursively visiting parent directories,
# looking for the file markers of the source root.
#
# Args:
#   $1 (str): The starting
get_source_tree_root() {
  # Check for the presence of files we expect in the source root.
  if [[ -f "$1/config.pbtxt" ]] && [[ -f "$1/WORKSPACE" ]]; then
    echo "$1"
  else
    if [[ "$1" == "/" ]]; then
      echo
    else
      echo $(get_source_tree_root $(realpath "$1/.."))
    fi
  fi
}

main() {
  local this_dir="$(dirname $(realpath $0))"
  local source_root="$(get_source_tree_root $this_dir)"

  if [[ -z "$source_root" ]]; then
    echo "Failed to find source root!"
    exit 1
  fi

  local this_dir_relpath="$(realpath --relative-to="$source_root" "$this_dir")"

  set -x
  # The --run_under argument runs the Jupyter server from the current directory
  # rather than the root of the build tree.
  bazel run -c opt --run_under="cd \"$this_dir\"; " "//$this_dir_relpath" -- $@
}
main $@
