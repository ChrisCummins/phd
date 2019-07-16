#!/usr/bin/env bash
#
# Export the shared code.
set -eux

tools/source_tree/export_source_tree \
    --target=//deeplearning/clgen/preprocessors:JavaRewriter,//deeplearning/deepsmith/harnesses:JavaDriver,//deeplearning/deepsmith/harnesses:JavaDriverTest,//deeplearning/clgen/preprocessors:java_test \
    --extra_file=.gitignore,experimental/deeplearning/deepsmith/java_fuzz/shared_code_README.md:README.md \
    --github_repo=ibm_deepsmith_shared_code
