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

# Root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"


main() {
  # Run from the artifact_evaluation root directory.
  cd "$ROOT"

  # If repository is cloned using https protocol, change the submodule
  # URLs to use https also.
  git remote -v | grep 'git@' &>/dev/null || \
      perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules

  # Checkout repository submodules.
  git submodule update --init

  # Bootstrap the phd repository.
  "$ROOT/tools/bootstrap.sh" | bash

  # Activate the phd virtual environment.
  test -f "$ROOT/.env"
  # Disable unbound variable errors, since .env checks if $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/.env"
  # Re-enable unbound variable errors.
  set -u

  # Generate in-tree files.
  "$ROOT/tools/protoc.sh"

  # Install the CLgen dependencies.
  cd "$ROOT/deeplearning/clgen"
  bash ./install-deps.sh

  # Configure and build CLgen.
  ./configure -b
  make
}
main $@
