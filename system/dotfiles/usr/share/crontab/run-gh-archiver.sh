#!/usr/bin/env bash
#
# Clone and update my GitHub repos locally.
#
# ****************************************************************************
# *                               Configuration                              *
# ****************************************************************************
export JOB_TIMEOUT=1800  # 30 minutes
export LMK="/usr/local/bin/lmk -e"
export LMK_TO="chrisc.101@gmail.com"


# ****************************************************************************
# *                                  Program                                 *
# ****************************************************************************
set -eux

if [[ -z "${1:-}" ]]; then
  # Run the section below script.
  $LMK --only-errors "timeout -s9 $JOB_TIMEOUT $0 --porcelain"
else
  cd /home/cec/phd
  bazel run //datasets/github/mirror_user -- \
      --gogs \
      --user=ChrisCummins \
      --dst=/home/cec/git/cec \
      --excludes=linux
fi
