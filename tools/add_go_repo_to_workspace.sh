#!/usr/bin/env bash
#
# Add a new go_repository() entry to the WORKSPACE file. E.g.:
#    ./tools/add_go_repo_to_workspace.sh github.com/stretchr/testify/assert
set -eux

bazel run //:gazelle -- update-repos $@
