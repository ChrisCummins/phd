#!/usr/bin/env bash
#
# Export the shared code.
set -eux

tools/source_tree/export_source_tree \
    --target=//deeplearning/clgen/preprocessors:JavaRewriter,//deeplearning/deepsmith/harnesses:JavaDriver \
    --extra_file=.gitignore \
    --github_repo=ibm_deepsmith_shared_code
