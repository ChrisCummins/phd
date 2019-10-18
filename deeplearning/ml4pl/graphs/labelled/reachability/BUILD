# Labelled graphs for reachability analysis.

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":reachability",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
        "//labm8:decorators",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "reachability",
    srcs = ["reachability.py"],
    deps = [
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "reachability_test",
    srcs = ["reachability_test.py"],
    deps = [
        ":make_reachability_dataset",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)
