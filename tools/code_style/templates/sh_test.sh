#!/usr/bin/env bash
#
# Smoke test for //TODO:${PACKAGE_NAME}/${NAME}
#
set -eux
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT


# Main entry point.
main() {
  # TODO: Implement.
}
main $@
