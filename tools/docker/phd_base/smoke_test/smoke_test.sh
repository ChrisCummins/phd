#!/usr/bin/env bash
#
# Smoke test for //tools/docker/phd_base:Dockerfile.
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
  cp tools/docker/phd_base/Dockerfile "$workdir"/Dockerfile
  docker build -t phd_base "$workdir"
  test $(docker run --entrypoint /usr/bin/id phd_base -un) = "docker"
}
main $@
