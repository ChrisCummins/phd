#!/usr/bin/env bash

#
# clean-git-tree - Remove redundant and dead git branches
#

set -ex

# Prune remote branches
git remote prune origin

# Prune local branches
git branch -r | awk '{print $1}' |
  egrep -v -f /dev/fd/0 <(git branch -vv | grep origin) |
  awk '{print $1}' | xargs git branch -d
