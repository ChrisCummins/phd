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
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "control_and_data_flow_graph_test",
    srcs = ["control_and_data_flow_graph_test.py"],
    deps = [
        ":control_and_data_flow_graph",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "control_and_data_flow_graph_regression_test",
    timeout = "long",
    srcs = ["control_and_data_flow_graph_regression_test.py"],
    data = ["//deeplearning/ml4pl/graphs/unlabelled/cdfg/regression_tests:bytecodes"],
    deps = [
        ":control_and_data_flow_graph",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "random_cdfg_generator",
    srcs = ["random_cdfg_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_and_data_flow_graph",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "random_cdfg_generator_test",
    srcs = ["random_cdfg_generator_test.py"],
    deps = [
        ":random_cdfg_generator",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
