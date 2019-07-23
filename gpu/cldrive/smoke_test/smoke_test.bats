#!/usr/bin/env bats
#
# Test running cldrive under oclgrind.
#
source labm8/sh/test.sh

tempdir="$(MakeTemporaryDirectory)"

setup() {
  mkdir -p "$tempdir"
  cat << EOF > "$tempdir/kernel.cl"
kernel void A(global int* a) {
  a[get_global_id(0)] *= 3;
}
EOF
}

teardown() {
  rm -rfv "$tempdir"
}

@test "fail with unknown arg" {
  run gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive --unknown_arg
  [ "$status" -eq 1 ]
}

@test "run clinfo" {
  run gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive --clinfo
  [ "$status" -eq 0 ]
}

@test "run with CSV output" {
  run gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive \
    --srcs="$tempdir/kernel.cl" \
    --num_runs=5
}

@test "run with protobuf output" {
  run gpu/oclgrind/oclgrind -- gpu/cldrive/cldrive \
    --srcs="$tempdir/kernel.cl" \
    --num_runs=5 \
    --output_format=pbtxt
}
