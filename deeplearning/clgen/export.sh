set -eux

destination=$HOME/tmp/clgen

targets=/tmp/targets.txt

bazel query '//deeplearning/clgen/...' > "$targets"

bazel run //tools/source_tree:export_source_tree -- \
    --targets=$(tr '\n' ',' < $targets | sed 's/,$//') \
    --destination=$destination

# cd $destination
# git checkout -- README.md
# rm gpu/cldrive/README.md
# rm configure
