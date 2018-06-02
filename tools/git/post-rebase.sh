#!/usr/bin/env bash
set -e

# Directory of the root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

main() {
  source "$ROOT/.env"
  git -C "$ROOT" submodule init --update
}

if [[ "$BRANCH_NAME" != '(no branch)' ]]; then
  main $@
fi
