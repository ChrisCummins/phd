#!/usr/bin/env bash
#
# Clone and update my GitHub repos locally.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
JOB_TIMEOUT=1800  # 30 min
GITHUB_USER=ChrisCummins
OUTDIR=~/git/github/
GH_ARCHIVER=/usr/local/bin/gh-archiver
# FIXME: clreduce repo contains broken reference, --exclude for now
GH_ARCHIVER_ARGS="--delete --exclude clreduce,distro,dos,staging --gogs --gogs-uid 2"
LMK="/usr/local/bin/lmk -e"

# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
    $LMK "timeout $JOB_TIMEOUT $0 --porcelain"
else
    $GH_ARCHIVER $GITHUB_USER -o $OUTDIR $GH_ARCHIVER_ARGS
fi
