#!/usr/bin/env bash
#
# Find and optionally remove conflicted Dropbox files.
# Feb 2017 Chris Cummins <chrisc.101@gmail.com>
#
# Usage:
#    ./find-dropbox-conflicts.sh       # list conflicted files
#    ./find-dropbox-conflicts.sh --rm  # remove conflicted files
#    DROPBOX_DIR=/tmp/dropbox ./find-dropbox-conflicts.sh  # custom dropbox location
#
usage() {
  echo "usage: $0 [--rm]" >&2
}

DROPBOX_DIR=${DROPBOX_DIR:-"$HOME/Dropbox"}

set -e

if [[ "$1" == "--help" ]]; then
  usage
elif [[ "$1" == "--rm" ]]; then
  find "$DROPBOX_DIR" -name '*conflicted copy*' -exec rm -v {} \;
elif [[ -n "$1" ]]; then
  usage
  exit 1
else
  find "$DROPBOX_DIR" -name '*conflicted copy*'
fi
