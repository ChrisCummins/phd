#!/usr/bin/env bash
#
# This script installs photolib.
#
set -eu

PREFIX=$HOME/.local

install_binary() {
  local target="$1"

  local source="util/photolib/$target.par"
  local destination="$PREFIX/bin/$(basename $target)"
  rm -fv "$destination"
  cp -v "$source" "$destination"
}

# Main entry point.
main() {
  mkdir -pv $PREFIX/bin
  install_binary photolib
  install_binary photolib-lint
}
main $@
