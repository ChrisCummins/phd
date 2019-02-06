#!/usr/bin/env bash
#
# Script to train and evaluate all combinations of model and dataset.
#
set -eux

# Path of input datasets. THIS SCRIPT DOES NOT GENERATE THESE FILES. Check the
# commands below to see what files it expects.
DATASETS_DIR="/var/phd/shared/docs/wip_graph/datasets"
# Path for output data.
OUTDIR_BASE="/var/phd/shared/docs/wip_graph/$(date +%s).$(hostname)"

# The outdir shouldn't exist.
test -d "$OUTDIR_BASE" && exit 1

bazel run //experimental/compilers/reachability/models:graph_model \
    -- --df="$DATASETS_DIR/opencl_devmap/amd.pkl" \
    --outdir="$OUTDIR_BASE/opencl_devmap/graph/amd" \
    --num_epochs=50 --v=1 \
    --experimental_force_num_processing_steps=50

bazel run //experimental/compilers/reachability/models:graph_model \
    -- --df="$DATASETS_DIR/opencl_devmap/nvidia.pkl" \
    --outdir="$OUTDIR_BASE/opencl_devmap/graph/nvidia" \
    --num_epochs=50 --v=1 \
    --experimental_force_num_processing_steps=50

bazel run //experimental/compilers/reachability/models:graph_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_ocl.pkl" \
    --outdir="$OUTDIR_BASE/reachability/graph/ocl" \
    --num_epochs=50 --v=1 \
    --experimental_force_num_processing_steps=50

bazel run //experimental/compilers/reachability/models:graph_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_linux.pkl" \
    --outdir="$OUTDIR_BASE/reachability/graph/linux" \
    --num_epochs=50 --v=1 \
    --experimental_force_num_processing_steps=50

bazel run //experimental/compilers/reachability/models:graph_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_linux_ocl.pkl" \
    --outdir="$OUTDIR_BASE/reachability/graph/linux_ocl" \
    --num_epochs=50 --v=1 \
    --experimental_force_num_processing_steps=50

bazel run //experimental/compilers/reachability/models:lstm_reachability_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_linux_ocl.pkl" \
    --outdir="$OUTDIR_BASE/reachability/lstm/linux_ocl" \
    ---v=1

bazel run //experimental/compilers/reachability/models:lstm_reachability_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_linux_ocl.pkl" \
    --outdir="$OUTDIR_BASE/reachability/lstm_neighbors_only/linux_ocl" \
    --neighbors_only --v=1

bazel run //experimental/compilers/reachability/models:lstm_reachability_model \
    -- --df="$DATASETS_DIR/reachability/synthetic_linux_ocl.pkl" \
    --outdir="$OUTDIR_BASE/reachability/zero_r/linux_ocl" \
    --zero_r --v=1

touch "$OUTDIR_BASE/done.txt"

echo "Finished all models" | lmk -
