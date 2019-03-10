#!/usr/bin/env bash
#
# Run args and return error.
set -eux
lmk --only-errors $@
