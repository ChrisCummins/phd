#!/usr/bin/env bash
set -ux

# requires $PLATFORM and $DEVICE
#
# example usage:
#     PLATFORM='NVIDIA CUDA' DEVICE='GeForce GTX 1080' bash ./dojob.sh

script="./clsmith-runprograms-cldrive.py"

while true; do
        $script "$PLATFORM" "$DEVICE" -g 1,1,1 -l 1,1,1 -s 256 -i arange
        sleep 1
        $script "$PLATFORM" "$DEVICE" -g 1,1,1 -l 1,1,1 -s 256 -i arange --no-opts
        sleep 1
        $script "$PLATFORM" "$DEVICE" -g 128,16,1 -l 32,1,1 -s 4096 -i arange
        sleep 1
        $script "$PLATFORM" "$DEVICE" -g 128,16,1 -l 32,1,1 -s 4096 -i arange --no-opts
        sleep 1
done
