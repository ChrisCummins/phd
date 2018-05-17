#!/usr/bin/env bash

# run.sh - Run the experiments.
#
# The experiments are written in Python in experiments.py for the
# implementation. This script only launches that one.
#
# Usage:
#
#     ./run.sh
#
set -eu


# Root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# The directory of this experiment.
WORKING_DIR="$ROOT/docs/2018_07_issta/artifact_evaluation/02_evaluate_harness"


main() {
  set +u
  OPENCL_DEVICE="$1"
  if [[ -z "$OPENCL_DEVICE" ]]; then
    echo "Please specify an OpenCL device number." >&2
    echo "List the available devices using: ./diagnose-me.sh" >&2
    exit 1
  fi
  set -u

  # Activate the phd virtual environment.
  test -f "$ROOT/.env"
  # Disable unbound variable errors, since .env checks if $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/.env"
  # Re-enable unbound variable errors.
  set -u

  mkdir -pv "$WORKING_DIR/output"

  # See file 'experiments.py' for the implementation.
  python "$WORKING_DIR/experiments.py" --testbed_id "$OPENCL_DEVICE"
}
main $@
