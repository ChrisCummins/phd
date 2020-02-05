#!/usr/bin/env bash
#
# Clean up merged local merged branches. Usage:
#
#     ./tools/git/delete_merged_branchs.sh
#
set -eux
if git branch --merged | egrep -v "(^\*|master|stable)"; then
  git branch --merged | egrep -v "(^\*|master|stable)" | xargs git branch -d
fi
git branch
