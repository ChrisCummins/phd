# Control and Data Flow Graphs.

py_library(
    name = "control_and_data_flow_graph",
    srcs = ["control_and_data_flow_graph.py"],
    data = [
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:dictionary",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:embeddings",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs:graph_iterators",
        "//deeplearning/ml4pl/graphs:graph_query",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//deeplearning/ml4pl/graphs/unlabelled/cg:call_graph",
        "//deeplearning/ncc:rgx_utils",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8:app",
        "//labm8:decorators",
        "//labm8:humanize",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "control_and_data_flow_graph_test",
    srcs = ["control_and_data_flow_graph_test.py"],
    deps = [
        ":control_and_data_flow_graph",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "make_unlabelled_graph_dataset",
    srcs = ["make_unlabelled_graph_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:database_exporters",
        "//deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8:app",
    ],
)
