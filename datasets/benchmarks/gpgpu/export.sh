# Export subtree.
set -eux

destination=$HOME/tmp/gpgpu_benchmarks

if [[ ! -d "$destination" ]]; then
  git clone git@github.com:ChrisCummins/gpgpu_benchmarks_early_export.git "$destination"
fi

bazel run //tools/source_tree:export_source_tree -- \
    --target=//datasets/benchmarks/gpgpu,//datasets/benchmarks/gpgpu:gpgpu_test \
    --destination="$destination"

cd $destination
