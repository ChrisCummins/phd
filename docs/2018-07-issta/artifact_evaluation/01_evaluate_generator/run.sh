#!/usr/bin/env bash

# run.sh - Train and sample a model on OpenCL examples.
#
# Usage:
#
#     ./run.sh
#
set -eu


# The artficat_evaluation root directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The directory of this experiment.
WORKING_DIR="$ROOT/docs/2018-07-issta/artifact_evaluation/01_evaluate_generator"


main() {
  # Activate the phd virtual environment.
  test -f "$ROOT/.env"
  # Disable unbound variable errors, since .env checks if $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/.env"
  # Re-enable unbound variable errors.
  set -u

  # Download and unpack the "tiny" corpus. The tiny corpus consists of 1000
  # anonymized files randomly selected from our GitHub corpus. The dataset
  # we used to train the model in the paper is much larger, but consequently
  # requires a longer training time.
  if [[ ! -f "$WORKING_DIR/data/tiny/1000.cl" ]]; then
    wget 'https://github.com/ChrisCummins/clgen/raw/master/clgen/test/data/tiny.tar.bz2' \
        -O "$WORKING_DIR/data/tiny.tar.bz2"
    mkdir -pv "$WORKING_DIR/data/tiny"
    tar xjf "$WORKING_DIR/data/tiny.tar.bz2" -C "$WORKING_DIR/data/tiny"
  fi

  # Configure the model which we will train. We have reduced the size of the
  # network so that it takes around 2 hours to train on a CPU using the "tiny"
  # corpus. We are going to train a 2 layer LSTM network with 256 neurons per
  # layer for 10 epochs. For the paper we trained a 2 layer LSTM network with
  # 512 neurons per layer for 50 epochs. You are free to change the values
  # used for the model. For example, you can train a 3 layer LSTM by changing
  # the line '"num_layers": 2' to '"num_layers": 3'. Please refer to the CLgen
  # source code for details:
  # https://github.com/ChrisCummins/clgen/blob/master/clgen/_model.py#L43-L63
  cat <<EOF > "$WORKING_DIR/data/model.json"
{
  "corpus": {
    "language": "opencl",
    "path": "$WORKING_DIR/data/tiny/corpus",
    "vocabulary": "greedy"
  },
  "train_opts": {
    "epochs": 10
  },
  "architecture": {
    "model_type": "lstm",
    "rnn_size": 256,
    "num_layers": 2
  }
}
EOF

  # Configure the sampling of new programs. For the purpose of demonstrating
  # our approach, we will generate 1000 new OpenCL kernels. You are free to
  # change the values used here. For example, to generate 5000 new kernels,
  # change the line '"min_samples": 1000', to '"min_samples": 5000'. Please
  # refer to the CLgen source code for details:
  # https://github.com/ChrisCummins/clgen/blob/master/clgen/_sampler.py#L47-L68
  cat <<EOF > "$WORKING_DIR/data/sampler.json"
{
  "sampler": {
    "min_samples": 1000,
    "static_checker": false
  },
  "kernels": {
    "language": "opencl",
    "max_length": 2000
  }
}
EOF

  # Run CLgen to automatically pre-process the corpus, train a model, and
  # sample from it.
  clgen sample "$WORKING_DIR/data/model.json" "$WORKING_DIR/data/sampler.json"

  # Once we are done, write the sampled OpenCL kernels to a directory for
  # inspection.
  rm -rf "$WORKING_DIR/output/generated_kernels"
  clgen -v db dump --dir --input-samples \
      $(clgen --sampler-dir "$WORKING_DIR/data/model.json" \
            "$WORKING_DIR/data/sampler.json")/kernels.db \
      "$WORKING_DIR/output/generated_kernels"
}
main $@
