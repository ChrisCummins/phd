#!/usr/bin/env bash
#
# Start a buildbot worker.

BUILDBOT_DIR="$HOME/buildbot"
WORKER="cc3"

set -eux

test -d "$BUILDBOT_DIR"/"$WORKER"
test -f "$BUILDBOT_DIR"/buildbot/bin/activate
cd "$BUILDBOT_DIR"
source buildbot/bin/activate
buildbot-worker start "$WORKER"
