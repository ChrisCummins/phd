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
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
  # Run from the artifact_evaluation root directory.
  cd "$ROOT"

  # Checkout the phd repository.
  if [ ! -d build/phd/.git ]; then
    mkdir -pv "$ROOT/build"
    git clone --depth 1 https://github.com/ChrisCummins/phd.git "$ROOT/build/phd"
  fi

  # Checkout phd repository submodules.
  cd "$ROOT/build/phd"
  perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules
  git submodule update --init
  cd "$ROOT"

  # Install the phd repository dependencies.
  if [ ! -f ./build/phd/.git/.env ]; then
    "$ROOT/build/phd/tools/bootstrap.sh" | bash
  fi

  # Activate the phd virtual environment.
  test -f "$ROOT/build/phd/.env"
  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/build/phd/.env"
  # Re-enable unbound variable errors.
  set -u

  # Generate in-tree files.
  "$ROOT/build/phd/tools/protoc.sh"

  # Build CLgen.
  cd "$ROOT/build/phd/deeplearning/clgen"
  ./configure -b
  make
}
main $@
