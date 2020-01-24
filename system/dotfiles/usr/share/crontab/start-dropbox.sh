#!/usr/bin/env bash
#
# Start Dropbox daemon.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
JOB_TIMEOUT=60 # 1 min
LMK="/usr/local/bin/lmk -e"
LMK_TO="chrisc.101@gmail.com"

DROPBOX=~/.local/bin/dropbox

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
  $LMK "timeout $JOB_TIMEOUT $0 --porcelain"
else
  $DROPBOX start
fi
