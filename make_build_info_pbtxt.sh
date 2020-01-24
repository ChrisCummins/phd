#!/usr/bin/env bash
#
# Script to generate a BuildInfo message instance.
#
# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -eu

workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

# Convert the status variables into bash array declarations and 'eval' them.
# Do this by adding a equals sign on the first whitespace and quoting the
# remainder of the line, e.g.
#   BUILD_TIMESTAMP 1552996705 -> BUILD_TIMESTAMP="1552996705"
sed 's/ /="/' bazel-out/volatile-status.txt | sed 's/$/"/' >"$workdir"/vars.sh
sed 's/ /="/' bazel-out/stable-status.txt | sed 's/$/"/' >>"$workdir"/vars.sh
source "$workdir"/vars.sh

cat <<EOF
# File: //config.proto
# Proto: BuildInfo
seconds_since_epoch: $BUILD_TIMESTAMP
unsafe_workspace: "$STABLE_UNSAFE_WORKSPACE"
host: "$STABLE_HOST"
user: "$STABLE_USER"
version: "$STABLE_VERSION"
repo_dirty: $STABLE_GIT_REPO_DIRTY
git_commit: "$STABLE_GIT_COMMIT_HASH"
git_branch: "$STABLE_GIT_BRANCH"
git_tracking: "$STABLE_GIT_REMOTE"
git_remote_url: "$STABLE_GIT_REMOTE_URL"
git_commit_author: "$STABLE_GIT_COMMIT_AUTHOR"
git_commit_seconds_since_epoch: $STABLE_GIT_COMMIT_TIMESTAMP
EOF
