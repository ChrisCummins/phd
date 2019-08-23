#!/usr/bin/env bash
#
# This script installs photolint.
#
set -eu

PREFIX=$HOME/.local

# Main entry point.
main() {
  mkdir -pv $PREFIX/bin
  rm -fv $PREFIX/bin/photolint
  cp -v util/photolib/photolib.par $PREFIX/bin/photolint
}
main $@
