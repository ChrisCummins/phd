#!/usr/bin/env bash
#
# Test that `generative_models` runs without catching fire.
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

research/cummins_2017_cgo/generative_model -- \
    --clgen_working_dir=$workdir \
    --clgen_corpus_dir=$corpus \
    --clgen_layer_size=8 \
    --clgen_max_sample_length=8 \
    --clgen_num_epochs=2 \
    --clgen_min_samples=5
