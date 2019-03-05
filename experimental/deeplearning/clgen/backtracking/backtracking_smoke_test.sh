#!/usr/bin/env bash
#
# Test that it backtracking produces 10 samples without blowing up.
#
corpus="$(mktemp -d)"
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$corpus"
  rm -fv "$corpus.sha1.txt"
  rm -rvf "$workdir"
}
trap cleanup EXIT

# Create corpus
cat << EOF > $corpus/a.txt
kernel void A(global int* a, const int b) {
  if (get_global_id(0) < b) {
    a[get_global_id(0)] *= 2;
  }
}
EOF

experimental/deeplearning/clgen/backtracking/backtracking \
    --clgen_working_dir $workdir \
    --clgen_corpus_dir=$corpus \
    --clgen_training_batch_size=1 \
    --clgen_training_sequence_length=2 \
    --clgen_layer_size=8 \
    --clgen_sample_sequence_length=10 \
    --clgen_min_sample_count=10 \
    --clgen_max_sample_length=8 \
    --clgen_num_epochs=1 \
    --clgen_sample_batch_size=1 \
    --experimental_clgen_backtracking_attempts=1
