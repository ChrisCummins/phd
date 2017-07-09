#!/usr/bin/env bash
set -eux


export PLATFORM_ID=$1
export DEVICE_ID=$2
export DEVICE_NAME=$3

lmk 'timeout -s 9 24h ./clreduce-run.sh $PLATFORM_ID $DEVICE_ID optimised $DEVICE_NAME'
lmk 'timeout -s 9 24h ./clreduce-run.sh $PLATFORM_ID $DEVICE_ID unoptimised $DEVICE_NAME'
