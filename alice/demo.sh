#!/usr/bin/env bash
# Proof of concept for remote exec.

# Input variables.
remote=cc1
#target=//deeplearning/deeptune/opencl/heterogeneous_mapping:models_test
#args="--test_color --test_durations=0"
target=//experimental/compilers/reachability/datasets:github_c
args="--cf=/var/phd/datasets/github/repos_by_lang/c.db --db=/tmp/c.db"
remote_python=/home/cec/.local/bin/python
remote_lmk=/usr/local/bin/lmk

# The script.

# Get the local commit hash. TODO: check for dirty repo.
commit=$(git rev-parse HEAD)

# Execute remote command.
ssh $remote <<EOF
cd ~/phd
git pull --rebase || exit 1
echo -n "Checking that commit hashes match ... "
if [[ "$commit" != "$(git rev-parse HEAD)" ]]; then
  echo " Local commit: $commit"
  echo "Remote commit: $(git rev-parse HEAD)"
  exit 1
fi
echo "ok"

$remote_python $remote_lmk 'bazel run $target -- $args'
EOF
