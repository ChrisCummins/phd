#!/usr/bin/env bash
set -eu

module="$1"

open="open"
opt=/usr/local/Cellar/llvm/8.0.1/bin/opt

for dotfile in $("$opt" -dot-cfg -dot-callgraph "$module" 2>&1 \
    | grep '^Writing ' \
    | sed -r "s/^Writing '(.+)'\.\.\.$/\1/"); do
  dotout="$(echo $dotfile | sed -r 's/^\.//')"
  pngout="$(echo $dotout | sed -r 's/\.dot$/.png/')"
  if [[ "$dotfile" != "$dotfile" ]]; then
    mv "$dotfile" "$dotout"
  fi
  dot "$dotout" -Tpng -o "$pngout";
  echo "$pngout"
done
