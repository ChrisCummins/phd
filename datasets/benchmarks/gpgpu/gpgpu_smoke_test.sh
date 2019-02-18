#!/usr/bin/env bash
#
# Test that `gpgpu` runs the dummy benchmark suite without catching fire.
#
set -eux

TMPDIR="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

datasets/benchmarks/gpgpu/gpgpu \
    --gpgpu_benchmark_suites=dummy_just_for_testing \
    --gpgpu_envs='Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' \
    --gpgpu_logdir="$TMPDIR"

ls "$TMPDIR"
