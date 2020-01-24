#!/usr/bin/env bash
set -eu

main() {
  test -f custom_spelling_words.txt
  for f in $(find . -name '*.tex'); do
    aspell --home-dir=. --personal=custom_spelling_words.txt -t -c $f
  done
}
main $@
