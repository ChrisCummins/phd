#!/usr/bin/env bash
#
# Start No-IP dynamic DNS daemon.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
JOB_TIMEOUT=60  # 1 min
LMK="/usr/local/bin/lmk -e"
LMK_TO="chrisc.101@gmail.com"

NOIP=/usr/local/bin/noip2

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
    $LMK "timeout $JOB_TIMEOUT $0 --porcelain"
else
    sudo $NOIP
fi
