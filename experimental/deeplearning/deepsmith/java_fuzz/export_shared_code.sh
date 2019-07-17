#!/usr/bin/env bash
#
# Export the shared code.
set -eux

targets=(
    //deeplearning/clgen/preprocessors:JavaRewriter
    //deeplearning/deepsmith/harnesses:JavaDriver
    //deeplearning/deepsmith/harnesses:JavaDriverTest
    //deeplearning/clgen/preprocessors:java_test
)

extra_files=(
    .gitignore
    experimental/deeplearning/deepsmith/java_fuzz/shared_code_README.md:README.md
)

# Join strings by a separator.
# Usage: join_by <separator> <<string> ...>
join_by() {
 local IFS="$1"
 shift
 echo "$*"
}

main() {
  targets_csv=$(join_by , "${targets[@]}")
  extra_files_csv=$(join_by , "${extra_files[@]}")

  tools/source_tree/export_source_tree \
      --targets="$targets_csv" \
      --extra_files="$extra_files_csv" \
      --github_repo=ibm_deepsmith_shared_code
}
main $@
