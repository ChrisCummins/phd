# Labelled graphs for data dependence.

py_library(
    name = "data_dependence",
    srcs = ["data_dependence.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_iterators",
        "//deeplearning/ml4pl/graphs:graph_query",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "data_dependence_test",
    srcs = ["data_dependence_test.py"],
    deps = [
        ":data_dependence",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_binary(
    name = "make_data_dependence_dataset",
    srcs = ["make_data_dependence_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_dependence",
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
