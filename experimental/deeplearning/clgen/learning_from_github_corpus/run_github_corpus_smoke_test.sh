#!/usr/bin/env bash
set -eux

TMP_CORPUS="$(mktemp -d)"
TMP_CACHED_RESULTS="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$TMP_CORPUS"
  rm -rf "$TMP_CACHED_RESULTS"
  rm -f "$TMP_CORPUS.sha1.txt"
}
trap cleanup EXIT

mkdir -pv "$TMP_CORPUS"

cat <<EOF > "$TMP_CORPUS/good.txt"
kernel void A(global int* a) {
  if (get_global_id(0) < 1000000) {
    a[get_global_id(0)] = 0;
  }
}
EOF

cat <<EOF > "$TMP_CORPUS/bad.txt"
This
file
contains
invalid
syntax!
EOF

experimental/deeplearning/clgen/learning_from_github_corpus/run_github_corpus \
  --github_kernels_dir="$TMP_CORPUS" --result_cache_dir="$TMP_CACHED_RESULTS"
