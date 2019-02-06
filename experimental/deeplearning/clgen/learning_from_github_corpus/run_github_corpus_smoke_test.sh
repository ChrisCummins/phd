#!/usr/bin/env bash
set -eux

TMP_CORPUS="/tmp/phd/corpus"
TMP_CACHED_RESULTS="/tmp/phd/cache"

mkdir -pv "$TMP_CORPUS"

echo <<EOF > "$TMP_CORPUS/good.txt"
kernel void A(global int* a) {
  a[get_global_id(0)] = 0;
}
EOF

echo <<EOF > "$TMP_CORPUS/bad.txt"
Syntax error!
EOF

experimental/deeplearning/clgen/learning_from_github_corpus/run_github_corpus \
  --github_kernels_dir="$TMP_CORPUS" --result_cache_dir="$TMP_CACHED_RESULTS"

# TODO(cec): Put in an exit handler.
rm -rf "$TMP_CORPUS"
rm -rf "$TMP_CACHED_RESULTS"
