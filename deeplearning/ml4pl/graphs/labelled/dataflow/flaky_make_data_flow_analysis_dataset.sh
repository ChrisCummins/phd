#!/usr/bin/env bash
#
# A wrapper around
# //deeplearning/ml4pl/graphs/labelled/dataflow:make_data_flow_analysis_dataset
# for running on flaky, heavily loaded systems.
#
# Usage:
#
#   bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:flaky_make_data_flow_analysis -- \
#       <make_data_flow_analysis_args...>
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
set -u

main() {
  i=0
  while true; do
    i=$((i + 1))
    echo "Beginning run $i of dataset generator"
    # Run the dataset generator for 30 minutes.
    if timeout -s9 1800 deeplearning/ml4pl/graphs/labelled/dataflow/make_data_flow_analysis_dataset \
      --max_instances=10000 \
      --vmodule='*'=3 \
      $@; then
      echo "Dataset generator terminated gracefully after $i iterations"
      break
    fi
    # Pause for a second so that the user can C-c twice to break the loop.
    if ! sleep 1; then
      break
    fi
  done
}
main $@
