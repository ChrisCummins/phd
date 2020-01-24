#!/usr/bin/env bash
#
# Clone and update my GitHub repos locally.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
JOB_TIMEOUT=60 # 1 min
LMK="/usr/local/bin/lmk -e"
LMK_TO="chrisc.101@gmail.com"

PIP_DB="https://www.pip-db.org"
WGET="wget"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
  $LMK "timeout $JOB_TIMEOUT $0 --porcelain"
else
  $WGET $PIP_DB -O /dev/null
fi
