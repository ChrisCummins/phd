#!/usr/bin/env bash
#
# Script to run a Jupyter notebook server in this directory.
#
# Usage:
#   bash run_notebooks.sh <notebooks_args>
#
# Where <notebooks_args> are the set of FLAGS defined in
# //notebooks:notebooks.py.
#
set -eu

# Find the root of the source tree by recursively visiting parent directories,
# looking for the file markers of the source root.
#
# Args:
#   $1 (str): The starting
get_source_tree_root() {
  # Check for the presence of files we expect in the source root.
  if [[ -f "$1/WORKSPACE" ]]; then
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
