#!/usr/bin/env bash
#
# Run the clinfo binary under oclgrind.
#
set -eux
third_party/oclgrind gpu/clinfo/clinfo
