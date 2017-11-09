#!/usr/bin/env bash
#
# Clone and update my GitHub repos locally.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
JOB_TIMEOUT=3600  # 1 hour
LMK="/usr/local/bin/lmk -e"
LMK_TO="chrisc.101@gmail.com"

GH_ARCHIVER="gh-archiver"
USERNAME="ChrisCummins"
CLONE_DIR="$HOME/git/cec"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
    $LMK "timeout -s9 $JOB_TIMEOUT $0 --porcelain"
else
    $GH_ARCHIVER $USERNAME --gogs -o $CLONE_DIR --exclude linux,paper-synthesizing-benchmarks,phd,staging
fi
