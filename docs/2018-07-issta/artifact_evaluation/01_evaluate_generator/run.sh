#!/usr/bin/env bash

# run.sh - Run the experiments.
#
# Usage:
#
#     ./run.sh
#
set -eu

# The artficat_evaluation root directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

main() {
  # Run from the artifact_evaluation root directory.
  cd "$ROOT"

  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  source "$ROOT/build/phd/.env"
  # Re-enable unbound variable errors.
  set -u

  # Download and unpack the "tiny" corpus. The tiny corpus consists of 1000
  # anonymized files randomly selected from our GitHub corpus.
  if [[ ! -f "$ROOT/01_evaluate_generator/data/tiny/1000.cl" ]]; then
    wget 'https://github.com/ChrisCummins/clgen/raw/master/clgen/test/data/tiny.tar.bz2' \
        -O "$ROOT/01_evaluate_generator/data/tiny.tar.bz2"
    mkdir -pv "$ROOT/01_evaluate_generator/data/tiny"
    tar xjf "$ROOT/01_evaluate_generator/data/tiny.tar.bz2" \
        -C "$ROOT/01_evaluate_generator/data/tiny"
  fi

  # Pre-process the corpus, train a model, and sample from it.
  clgen sample \
      "$ROOT/01_evaluate_generator/data/model.json" \
      "$ROOT/01_evaluate_generator/data/sampler.json"

  # Dump the generated kernels to a directory.
  clgen -v db dump $(clgen --sampler-dir \
      "$ROOT/01_evaluate_generator/data/model.json" \
      "$ROOT/01_evaluate_generator/data/sampler.json")/kernels.db \
      "$ROOT/01_evaluate_generator/run/generated_kernels" --dir --input-samples
}
main $@
