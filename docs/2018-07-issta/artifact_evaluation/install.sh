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

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
  cd "$DIR"

  # Checkout the phd repository.
  if [ ! -d build/phd/.git ]; then
    mkdir build
    git clone --recursive https://XXXX.git ./build/phd
    cd build/phd
    cd ../..
  fi

  # Install the phd repository dependencies.
  if [ ! -f ./build/phd/.git/.env ]; then
    ./build/phd/tools/bootstrap.sh | bash
  fi

  # Activate the phd virtual environment.
  source ./build/phd/.env

  # Generate in-tree files.
  ./build/phd/tools/protoc.sh
}

main $@
