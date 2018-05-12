#!/usr/bin/env bash
#
# rm-dsstore - Remove OS X crud files
#
set -e

if [[ -z "$1" ]]; then
  dir="."
else
  dir="$1"
fi

set -x
find "$dir" \( -name '._*' -o -name '.DS_Store' \) -exec rm -v {} \;
