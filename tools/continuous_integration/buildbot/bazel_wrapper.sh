#!/usr/bin/env bash
#
# Run bazel but don't propogate an error return.
set -x
bazel $@
echo "'bazel $@' returned $?"
