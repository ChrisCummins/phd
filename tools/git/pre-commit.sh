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

  git -C "$ROOT" fetch origin

  COMMITS_BEHIND_UPSTREAM="$(git rev-list --left-right --count origin/master...@ | cut -f1)"

  if [[ "$COMMITS_BEHIND_UPSTREAM" > 0 ]]; then
    echo "Current branch is $COMMITS_BEHIND_UPSTREAM commits behind origin/master, creating stash ..." >&2
    git -C "$ROOT" stash
    echo "Rebasing ..."
    git -C "$ROOT" pull --rebase
    echo "Popping stash ..."
    git -C "$ROOT" stash pop --index
    echo 'Inspect and re-run git commit'
    exit 1
  fi
}

if [[ "$BRANCH_NAME" != '(no branch)' ]]; then
  main $@
fi
