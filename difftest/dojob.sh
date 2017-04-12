#!/usr/bin/env bash
set -ux

# requires $SCRIPT, $PLATFORM, and $DEVICE
#
# example usage:
#     SCRIPT='clsmith-run-cldrive.py' PLATFORM='NVIDIA CUDA' DEVICE='GeForce GTX 1080' bash ./dojob.sh
SCRIPT="$1"
PLATFORM="$2"
DEVICE="$3"


while true; do
        ./"$SCRIPT" "$PLATFORM" "$DEVICE" -g 1,1,1 -l 1,1,1 -s 256 -i arange
        sleep 1
        ./"$SCRIPT" "$PLATFORM" "$DEVICE" -g 1,1,1 -l 1,1,1 -s 256 -i arange --no-opts
        sleep 1
        ./"$SCRIPT" "$PLATFORM" "$DEVICE" -g 128,16,1 -l 32,1,1 -s 4096 -i arange
        sleep 1
        ./"$SCRIPT" "$PLATFORM" "$DEVICE" -g 128,16,1 -l 32,1,1 -s 4096 -i arange --no-opts
        sleep 1
done
