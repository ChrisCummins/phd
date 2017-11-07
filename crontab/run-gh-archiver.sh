#!/usr/bin/env bash
#
# Run gh-archiver.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
export JOB_TIMEOUT=1800  # 30 minutes
export LMK="/usr/local/bin/lmk -e"
export LMK_TO="chrisc.101@gmail.com"

export GH_ARCHIVER="gh-archiver"
export GH_USERNAME="ChrisCummins"
export GOGS_CLONE_DIR="/home/cec/git/cec"


# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

$LMK --only-errors "timeout -s9 $JOB_TIMEOUT $GH_ARCHIVER $GH_USERNAME --gogs -o $GOGS_CLONE_DIR"
