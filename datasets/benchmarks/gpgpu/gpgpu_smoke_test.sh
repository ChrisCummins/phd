#!/usr/bin/env bash
#
# Test that `gpgpu` runs the dummy benchmark suite without catching fire.
#
set -eux

TMP_LOGDIR="/tmp/phd/gpgpu"

datasets/benchmarks/gpgpu/gpgpu \
    --gpgpu_benchmark_suites=dummy_just_for_testing \
    --gpgpu_envs='Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' \
    --gpgpu_logdir="$TMP_LOGDIR"

ls "$TMP_LOGDIR"

rm -rf "$TMP_LOGDIR"
