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

  # Bootstrap the phd repository if necessary.
  test -f "$ROOT/.env" || "$ROOT/tools/bootstrap.sh"

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
  yes | bash ./install-deps.sh

  # Configure and build CLgen.
  ./configure -b
  make
}
main $@
