#!/usr/bin/env bash
set -eux

zero_r() {
  local gpu="$1"
  local test="$2"
  local val="$3"

  deeplearning/ml4pl/models/ggnn/ggnn \
      --log_db='file:///var/phd/db/cc1.mysql?ml4pl_logs?charset=utf8' \
      --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_'"$gpu"'?charset=utf8' \
      --num_epochs=100 \
      --graph_state_dropout_keep_prob=.5 \
      --output_layer_dropout_keep_prob=.5 \
      --edge_weight_dropout_keep_prob=.5 \
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
