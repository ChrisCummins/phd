#!/usr/bin/env bash
set -eu

benchmarks=(
    # amd-app-sdk-3.0
    npb-3.3
    nvidia-4.2
    parboil-0.2
    polybench-gpu-1.0
    rodinia-3.1
    # shoc-1.1.5
)

for benchmark in ${benchmarks[@]}; do
    echo "$(tput bold)$benchmark$(tput sgr0) ..."
    cd $benchmark
    $0 $@
    cd ..
done
