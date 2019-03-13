#!/usr/bin/env bash
#
# Start the buildbot master.

BUILDBOT_DIR="$HOME/buildbot"

set -eux

test -d "$BUILDBOT_DIR"/master
test -f "$BUILDBOT_DIR"/buildbot/bin/activate
cd "$BUILDBOT_DIR"
source buildbot/bin/activate
buildbot start master
