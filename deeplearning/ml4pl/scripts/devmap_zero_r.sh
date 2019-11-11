#!/usr/bin/env bash
set -eux

zero_r() {
  local gpu="$1"
  local test="$2"
  local val="$3"

  sleep 1  # For unique run ID.
  deeplearning/ml4pl/models/zero_r/zero_r \
      --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_'"$gpu"'?charset=utf8' \
      --log_db='file:///var/phd/db/cc1.mysql?ml4pl_logs?charset=utf8' \
      --working_dir='/var/phd/shared/ml4pl' \
      --num_epochs=1 \
      --batch_size=100000 \
      --batch_scores_averaging_method=binary \
      --test_group="$test" --val_group="$val"
}

main() {
  zero_r nvidia 0 1
  zero_r nvidia 1 2
  zero_r nvidia 2 3
  zero_r nvidia 3 4
  zero_r nvidia 4 5
  zero_r nvidia 5 6
  zero_r nvidia 6 7
  zero_r nvidia 7 8
  zero_r nvidia 8 9
  zero_r nvidia 9 0

  zero_r amd 0 1
  zero_r amd 1 2
  zero_r amd 2 3
  zero_r amd 3 4
  zero_r amd 4 5
  zero_r amd 5 6
  zero_r amd 6 7
  zero_r amd 7 8
  zero_r amd 8 9
  zero_r amd 9 0
}
main $@
