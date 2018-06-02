#!/usr/bin/env bash
set -e

# Directory of the root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The name of the current branch, or '(no branch)' if currently headless (such
# as while rebasing).
BRANCH_NAME="$(git -C "$ROOT" branch | grep '*' | sed 's/* //')"

main() {
  source "$ROOT/.env"
  "$PHD/tools/buildifier.sh"

  if [[ "$(git rev-list --left-right --count master...@ | cut -f1)" > 0 ]]; then
    echo 'fatal: Current branch is ahead' >&2
    exit 1
  fi
}

if [[ "$BRANCH_NAME" != '(no branch)' ]]; then
  main $@
fi
