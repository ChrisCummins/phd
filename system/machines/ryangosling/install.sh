#!/usr/bin/env bash
#
# This script installs the ryangosling machines config into ~/.local.
#
# The files installed in this script are referenced by ~/.zsh/ryangosling.zsh.
#
set -eu

PREFIX=$HOME/.local
HOSTNAME=$(hostname | sed 's/\.local$//')

set -x

# Main entry point.
main() {
  mkdir -pv $PREFIX/bin $PREFIX/var/machines
  rm -f $PREFIX/bin/machines
  cp system/machines/machine.par $PREFIX/bin/machines
  cp system/machines/ryangosling/$HOSTNAME.pbtxt \
    $PREFIX/var/machines/ryangosling.pbtxt
  cp system/machines/ryangosling/$HOSTNAME.zsh \
    $HOME/.zsh/ryangosling.zsh
}
main $@
