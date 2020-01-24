#!/usr/bin/env bash

set -eu

files="$(find include/ustl -type f)"

(
  echo "FILE LINES TODOs"
  for file in $files; do
    echo -n "$file "
    set +e
    wc -l "$file" | awk '{printf "%d ", $1;}'
    grep -c 'TODO:' "$file"
    set -e
  done
) | sort -nk 3 | column -t
