#!/usr/bin/env bash
#
# This script installs dpack into ~/.local.
#
set -eux

PREFIX=$HOME/.local

# Main entry point.
main() {
  mkdir -pv $PREFIX/bin
  rm -f $PREFIX/bin/dpack
  cp system/dpack/dpack.par $PREFIX/bin/dpack
}
main $@
