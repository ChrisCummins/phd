#!/usr/bin/env bash
# Run the me_db notebook server.

bazel build "//datasets/me_db/database"
bazel-bin/datasets/me_db/notebooks/notebooks
