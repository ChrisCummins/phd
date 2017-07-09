#!/usr/bin/env bash
set -eux

cd ~/src/project_b/lib/clreduce
. .env

export PLATFORM_ID=$1
export DEVICE_ID=$2
export DEVICE_NAME=$3

timeout -s 9 24h lmk "bash ./run.sh $PLATFORM_ID $DEVICE_ID optimised ~/src/project_b/data/clreduce/$DEVICE_NAME/on/$(date '+%Y-%m-%d-%H:%M')"
timeout -s 9 24h lmk "bash ./run.sh $PLATFORM_ID $DEVICE_ID unoptimised ~/src/project_b/data/clreduce/$DEVICE_NAME/off/$(date '+%Y-%m-%d-%H:%M')"
