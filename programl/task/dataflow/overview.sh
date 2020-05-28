#!/usr/bin/env bash

help() {
  cat <<EOF
overview: Print an overview of dataset directory.

Usage:

  overview <dataset_path>

Where <dataset_path> is the root directory of the dataflow dataset.
EOF
}

set -euo pipefail

# Overview of dataset stats.
# Must be called from the root of the dataflow dataset.
overview() {
  echo "=========================="
  echo "#. runs: " $(find -L logs -mindepth 3 -maxdepth 3 -type d | wc -l)
  echo
  echo "Epochs: "
  for run in $(find -L logs -mindepth 3 -maxdepth 3 -type d | sort); do
    echo "    $(find $run/epochs -type f | wc -l) $run"
  done

  echo
  echo "=========================="
  echo "Dataset job logs:"
  find labels -mindepth 1 -maxdepth 1 -type f | sort | xargs wc -l | sed 's/^/    /'

  echo
  echo "=========================="
  echo "Directory sizes:"
  for dir in graphs $(find labels -mindepth 1 -maxdepth 1 -type d | sort); do
    echo "    $(find $dir -type f | wc -l) $dir"
  done

  echo
  echo "=========================="
  echo "Splits:"
  for dir in train val test; do
    echo "    $(find $dir -type l | wc -l) $dir"
  done

  echo
  echo "=========================="
  echo "Graph sources:"
  find graphs -type f | grep -v poj104 | grep -v github | cut -d'.' -f1 | sort | uniq -c
  find graphs -type f -name 'poj104*' | cut -d'_' -f1 | sort | uniq -c
  find graphs -type f -name 'github*' | cut -d'.' -f1,3 | sort | uniq -c

  if [[ -d ll ]]; then
    echo
    echo "=========================="
    echo "IR line counts:"
    echo "BLAS          $(find ll -type f -name 'blas*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "Linux         $(find ll -type f -name 'linux*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "OpenCL        $(find ll -type f -name 'opencl*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "OpenCV        $(find ll -type f -name 'opencv*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "POJ-104       $(find ll -type f -name 'poj104*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "TensorFlow    $(find ll -type f -name 'tensorflow*.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "Github C      $(find ll -type f -name 'github.*.c.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "Github OpenCL $(find ll -type f -name 'github.*.cl.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
    echo "Github Swift  $(find ll -type f -name 'github.*.swift.ll' | xargs wc -l | tail -n1 | sed -r 's/^ +//' | cut -d' ' -f1)"
  fi
}

main() {
  set +u
  if [[ -z "$1" ]]; then
    help >&2
    exit 1
  fi
  set -u

  if [[ "$1" == "--help" ]]; then
    help
    exit 1
  fi

  if [[ ! -d "$1" ]]; then
    echo "Directory not found: $1" >&2
    exit 1
  fi

  cd "$1"
  shift
  set +u
  if [[ -n "$1" ]]; then
    help >&2
    exit 1
  fi
  set -u

  overview
}
main $@
