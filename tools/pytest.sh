#!/usr/bin/env bash

# pytest.sh - Run pytest on *_test.py files.
#
# Usage:
#
#     ./pytest.sh
#
set -eu

# The root of the repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

main() {
  cd "$ROOT"
  for file in $(git ls-files | grep '_test.py$'); do
    echo $file
    python $file
  done
}
main $@
