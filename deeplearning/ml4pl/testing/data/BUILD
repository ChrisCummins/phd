# Test data for //deeplearning/ml4pl.

# A random 100 unlabelled graphs, as a compressed tarball of pickled networkx
# graphs.
#
# Generated using:
#     bazel run //deeplearning/ml4pl/graphs:copy_database -- \
#        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus' \
#        --output_dir=/tmp/graphs \
#        --max_rows=100 \
#        --max_node_count=8000
filegroup(
    name = "100_unlabelled_networkx_graphs",
    testonly = 1,
    srcs = ["100_unlabelled_networkx_graphs.tar.bz2"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
)

# A random 100 unlabelled networkx graphs, as a compressed SQLite graph_database.
#
# Generated using:
#     bazel run //deeplearning/ml4pl/graphs:copy_database -- \
#        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus' \
#        --output_db='sqlite:////tmp/graphs.db' \
#        --max_rows=100 \
#        --max_node_count=8000
filegroup(
    name = "100_unlabelled_networkx_graphs_db",
    testonly = 1,
    srcs = ["100_unlabelled_networkx_graphs.db.tar.bz2"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
)
