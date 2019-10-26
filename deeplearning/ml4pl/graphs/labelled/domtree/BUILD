# Labelled graphs for dominator trees.

py_library(
    name = "dominator_tree",
    srcs = ["dominator_tree.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_iterators",
        "//deeplearning/ml4pl/graphs:graph_query",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "dominator_tree_test",
    srcs = ["dominator_tree_test.py"],
    deps = [
        ":dominator_tree",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_binary(
    name = "make_dominator_tree_dataset",
    srcs = ["make_dominator_tree_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dominator_tree",
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
