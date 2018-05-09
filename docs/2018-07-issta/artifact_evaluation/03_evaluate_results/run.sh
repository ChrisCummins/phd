#!/usr/bin/env bash

# run.sh - Run the experiments.
#
# Usage:
#
#     ./run.sh
#
set -eu


main() {
  # Run from the artifact_evaluation root directory.
  cd "$(dirname "${BASH_SOURCE[0]}")/.."

  # Activate the phd virtual environment.
  source ./build/phd/.env

  # Run the python implementation script.
  python ./03_evaluate_results/experiments.py
}
main $@
