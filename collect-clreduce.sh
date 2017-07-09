set -eux
export PLATFORM_ID=$1
export DEVICE_ID=$2
export DEVICE_NAME=$3

cd ~/src/project_b/lib/clreduce
while true; do
    lmk "bash ./run.sh $PLATFORM_ID $DEVICE_ID optimised ~/src/project_b/data/clreduce/$DEVICE_NAME/on/$(date '+%Y-%m-%d-%H:%M')"
    lmk "bash ./run.sh $PLATFORM_ID $DEVICE_ID unoptimised ~/src/project_b/data/clreduce/$DEVICE_NAME/off/$(date '+%Y-%m-%d-%H:%M')"
done
