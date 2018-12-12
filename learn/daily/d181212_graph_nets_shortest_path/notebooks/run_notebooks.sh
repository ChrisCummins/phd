#!/usr/bin/env bash
# Script to run Jupyter notebook server from root of the source tree.
set -eux

# The --run_under argument runs the Jupyter server from the root of the source
# tree rather than the root of the build tree.
bazel run --run_under="cd \"$(pwd)\"; " //learn/daily/d181212_graph_nets_shortest_path/notebooks
