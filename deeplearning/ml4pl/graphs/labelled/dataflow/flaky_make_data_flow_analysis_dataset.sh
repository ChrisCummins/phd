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
set -u

main() {
  i=0
  while true; do
    i=$((i+1))
    echo "Beginning run $i of dataset generator"
    # Run the dataset generator for 30 minutes.
    timeout -s9 1800 deeplearning/ml4pl/graphs/labelled/dataflow/make_data_flow_analysis_dataset \
        --max_instances=10000 \
        --vmodule='*'=3 \
        $@
    # If the dataset generator completed with a zero return code, terminate.
    if $@ ; then
      echo "Dataset generator terminated gracefully after $i iterations"
      break
    fi
    # Pause for a second so that the user can C-c twice to break the loop.
    if ! sleep 1 ; then
      break
    fi
  done
}
main $@
