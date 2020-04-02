#!/usr/bin/env bash
#
# Benchmark script for //deeplearning/ml4pl/graphs/llvm2graph.
#
# Extracts a test set of LLVM-IRs and runs llvm2graph on each of them in-turn.
# Prints a tab-separated list of IR file names and llvm2graph runtimes (in
# seconds). After processing all files, it prints a one line summary of results.
#
# Usage:
#
#     $ bazel run //deeplearning/ml4pl/graphs/llvm2graph:llvm2graph_benchmark
#     10059.ll  0.733
#     100969.ll 0.017
#     101540.ll 0.015
#     101665.ll 0.107
#     102046.ll 0.017
#     102603.ll 0.027
#     102963.ll 0.026
#     103115.ll 0.010
#     ...
#     Processed 500 LLVM-IRs in 69 seconds, 138.000 ms / file
#
# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null ||
  source "$0.runfiles/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  {
    echo >&2 "ERROR: cannot find $f"
    exit 1
  }
f=
# --- end app init ---

set -eu

LLVM2GRAPH="$(DataPath phd/deeplearning/ml4pl/graphs/llvm2graph/llvm2graph)"
LLVM_IR_TAR="$(DataPath phd/deeplearning/ml4pl/testing/data/llvm_ir.tar.bz2)"

main() {
  # Extract the test files.
  tar -xjf "$LLVM_IR_TAR" -C "$TEST_TMPDIR"
  num_files="$(ls $TEST_TMPDIR | wc -l)"

  # Reset the bash internal timer and increase the precision of `time` builtin.
  TIMEFORMAT='%3R'
  SECONDS=0
  # Process each of the test files.
  for f in "$TEST_TMPDIR"/*.ll; do
    echo -en "$(basename $f)\t"
    time "$LLVM2GRAPH" "$f" >/dev/null
  done

  # Print a summary of results.
  ms_per_file="$(echo "$SECONDS * 1000 / $num_files.0" | bc -l | xargs printf %.3f)"
  echo "Processed $(ls $TEST_TMPDIR | wc -l) LLVM-IRs in $SECONDS seconds, $ms_per_file ms / file"
}
main
