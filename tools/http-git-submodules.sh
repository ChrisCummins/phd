#!/usr/bin/env bash

help() {
  echo "usage: $0"
  echo
  echo "Replace git protocol submodules with https."
}

test -z "$1" || {
  help
  exit 1
}

set -eu

test -f .gitmodules

if sed --version &>/dev/null; then
  sed -i 's,git@github.com:,https://github.com/,' .gitmodules
else
  sed -i '' 's,git@github.com:,https://github.com/,' .gitmodules
fi

cat .gitmodules
