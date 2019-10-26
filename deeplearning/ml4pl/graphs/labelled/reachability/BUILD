# Labelled graphs for reachability analysis.

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":reachability",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/bytecode:splitters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:database_exporters",
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

py_test(
    name = "make_reachability_dataset_test",
    srcs = ["make_reachability_dataset_test.py"],
    deps = [
        ":make_reachability_dataset",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "reachability",
    srcs = ["reachability.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_query",
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
