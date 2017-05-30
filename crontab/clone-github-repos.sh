#!/usr/bin/env bash
#
# Clone and update my GitHub repos locally.
#
set -eu

# maximum runtime in seconds
JOB_TIMEOUT=1800  # 30 min

GITHUB_USER=ChrisCummins
OUTDIR=~/src/GitHub/ChrisCummins

set +e
# FIXME: clreduce repo contains broken reference, --exclude for now
timeout $JOB_TIMEOUT /usr/local/bin/gh-archiver $GITHUB_USER -o $OUTDIR --delete --exclude clreduce,distro
ret=$?
test $ret != 124 || { echo "timeout after $JOB_TIMEOUT seconds"; exit $ret; }
test $ret = 0 || exit $ret
