#!/usr/bin/env bash
#
# Run PlexPy server.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
LMK="/usr/local/bin/lmk -e"
LMK_TO="chrisc.101@gmail.com"

PLEXPY_DIR=/opt/src/plexpy
PLEXPY="$PLEXPY_DIR/PlexPy.py"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
  $LMK "$0 --porcelain"
else
  cd "$PLEXPY_DIR"
  "$PLEXPY"
fi
