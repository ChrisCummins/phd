#!/usr/bin/env bash
#
# A smoke test to ensure that target runs without crashing. This doesn't test
# the results of execution, other than the return code.

set -eux

deeplearning/ncc/train_task_devmap \
    --v=1 \
    --num_epochs=1 \
    --embeddings_file=deeplearning/ncc/published_results/emb.p \
    --vocabulary_zip_path=deeplearning/ncc/published_results/vocabulary.zip
