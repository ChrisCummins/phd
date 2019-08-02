#!/usr/bin/env bash
#
# This script installs the ryangosling machines config into ~/.local.
#
# The files installed in this script are referenced by ~/.zsh/ryangosling.zsh.
#
set -eux

PREFIX=$HOME/.local

# Main entry point.
main() {
  mkdir -pv $PREFIX/bin $PREFIX/var/machines
  rm -f $PREFIX/bin/machines
  cp system/machines/machine.par $PREFIX/bin/machines
  cp system/machines/ryangosling/ryangosling.pbtxt \
    $PREFIX/var/machines/ryangosling.pbtxt
}
main $@
