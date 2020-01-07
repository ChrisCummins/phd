#!/usr/bin/env bash
#
# A wrapper script for invoking a platform-specific oclgrind binary.
#
# Usage:
#
#    bazel run //third_party:oclgrind -- <oclgrind_args>
#
set -eu
if [[ -f external/oclgrind_linux/bin/oclgrind ]]; then
  external/oclgrind_linux/bin/oclgrind $@
elif [[ -f external/oclgrind_mac/bin/oclgrind ]]; then
  external/oclgrind_mac/bin/oclgrind $@
else
  echo "oclgrind not found!" >&2;
  exit 1
fi
