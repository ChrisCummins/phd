#!/usr/bin/env bash

# install.sh - Prepare the toolchain.
#
# This may take some time to complete. This may be run repeatedly, things won't
# be installed twice.
#
# Usage:
#
#     ./install.sh
#
set -eux

# This directory.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
  # Run from the artifact_evaluation root directory.


  # Checkout the phd repository.
  if [ ! -d build/phd/.git ]; then
    mkdir -pv "$DIR/build"
    git clone --depth 1 https://github.com/ChrisCummins/phd.git "$DIR/build/phd"
  fi

  # Checkout phd repository submodules.
  cd "$DIR/build/phd"
  perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules
  git submodule update --init
  cd "$DIR"

  # Install the phd repository dependencies.
  if [ ! -f ./build/phd/.git/.env ]; then
    "$DIR/build/phd/tools/bootstrap.sh" | bash
  fi

  # Activate the phd virtual environment.
  test -f "$DIR/build/phd/.env"
  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  source "$DIR/build/phd/.env"
  # Re-enable unbound variable errors.
  set -u

  # Generate in-tree files.
  "$DIR/build/phd/tools/protoc.sh"

  # Build CLgen.
  cd "$DIR/build/phd/deeplearning/clgen"
  ./configure -b
  make
}
main $@
