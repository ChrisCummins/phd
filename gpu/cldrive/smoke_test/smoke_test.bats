#!/usr/bin/env bats
#
# Test running cldrive under oclgrind.
#
source labm8/sh/test.sh

OCLGRIND="$(DataPath phd/third_party/oclgrind/oclgrind)"
CLDRIVE="$(DataPath phd/gpu/cldrive/cldrive)"

setup() {
  cat << EOF > "$BATS_TMPDIR/kernel.cl"
kernel void A(global int* a) {
  a[get_global_id(0)] *= 3;
}
EOF
}

@test "fail with unknown arg" {
  run "$OCLGRIND" "$CLDRIVE" --unknown_arg
  [ "$status" -eq 1 ]
}

@test "run clinfo" {
  run "$OCLGRIND" "$CLDRIVE" --clinfo
  [ "$status" -eq 0 ]
}

@test "run with CSV output" {
  run "$OCLGRIND" "$CLDRIVE" \
    --srcs="$BATS_TMPDIR/kernel.cl" \
    --num_runs=5
}

@test "run with protobuf output" {
  run "$OCLGRIND" "$CLDRIVE" \
    --srcs="$BATS_TMPDIR/kernel.cl" \
    --num_runs=5 \
    --output_format=pbtxt
}
