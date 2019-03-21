#!/usr/bin/env bash
#
# Test that cldrive works without blowing up.
#
set -eux

workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT

cat << EOF > "$workdir"/kernel.cl
kernel void A(global int* a) {
  a[get_global_id(0)] *= 3;
}
EOF

# Run clinfo.
gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive --clinfo

# Run with CSV output.
gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive \
  --srcs="$workdir"/kernel.cl \
  --num_runs=10

# Run with protobuf output.
gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive \
  --srcs="$workdir"/kernel.cl \
  --num_runs=5 \
  --output_format=pbtxt
