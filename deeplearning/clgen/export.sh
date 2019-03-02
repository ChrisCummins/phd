set -eux

destination=$HOME/tmp/clgen

targets=/tmp/targets.txt

bazel query 'kind(cc_test,//deeplearning/clgen/...)' > "$targets"
bazel query 'kind(py_test,//deeplearning/clgen/...)' >> "$targets"
echo '//deeplearning/clgen' >> "$targets"

bazel run //tools/source_tree:export_source_tree -- \
    --target=$(tr '\n' ',' < $targets | sed 's/,$//') \
    --destination=$destination

# cd $destination
# git checkout -- README.md
# rm gpu/cldrive/README.md
# rm configure
