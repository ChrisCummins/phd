#!/usr/bin/env bash
#
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
sed 's/ /="/' bazel-out/volatile-status.txt | sed 's/$/"/' > "$workdir"/vars.sh
sed 's/ /="/' bazel-out/stable-status.txt | sed 's/$/"/' >> "$workdir"/vars.sh
source "$workdir"/vars.sh

cat << EOF
# File: //config/proto/config.proto
# Proto: BuildInfo
seconds_since_epoch: $BUILD_TIMESTAMP
host: "$STABLE_HOST"
user: "$STABLE_USER"
git_hash: "$STABLE_GIT_COMMIT_HASH"
repo_dirty: $STABLE_GIT_REPO_DIRTY
git_branch: "$STABLE_GIT_BRANCH"
git_remote: "$STABLE_GIT_REMOTE"
git_remote_url: "$STABLE_GIT_REMOTE_URL"
git_commit_author: "$STABLE_GIT_COMMIT_AUTHOR"
git_commit_seconds_since_epoch: $STABLE_GIT_COMMIT_TIMESTAMP
EOF
