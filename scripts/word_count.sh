#!/usr/bin/env bash
set -eu

main() {
  test -d .git
  find . -name '*.tex' | xargs texcount 2>/dev/null | sed '0,/^Total$/d'
}
main $@
