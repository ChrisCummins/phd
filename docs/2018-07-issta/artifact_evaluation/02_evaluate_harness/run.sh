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


# The artficat_evaluation root directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The directory of this experiment.
WORKING_DIR="$ROOT/docs/2018-07-issta/artifact_evaluation/01_evaluate_generator"


main() {
  # Activate the phd virtual environment.
  test -f "$ROOT/.env"
  # Disable unbound variable errors, since .env checks if $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/.env"
  # Re-enable unbound variable errors.
  set -u

  mkdir -pv "$WORKING_DIR/run"

  # See this file for the experiment implementation.
  python "$WORKING_DIR/experiments.py"
}
main $@
