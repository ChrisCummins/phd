#!/usr/bin/env bash
set -ux

cd ~/src/project_b/lib/clreduce
. .env

export PLATFORM_ID=$1
export DEVICE_ID=$2
export OPTIMISED=$3
export DEVICE_NAME=$4

while true; do
    bash ./run.sh $PLATFORM_ID $DEVICE_ID $OPTIMISED ~/src/project_b/data/clreduce/$DEVICE_NAME/$OPTIMISED/$(date '+%Y-%m-%d-%H:%M')
done
