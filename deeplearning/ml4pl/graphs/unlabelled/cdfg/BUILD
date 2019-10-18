# Control and Data Flow Graphs.

py_library(
    name = "control_and_data_flow_graph",
    srcs = ["control_and_data_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/networkx",
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
