# Labelled graphs for dominator trees.

py_library(
    name = "dominator_tree",
    srcs = ["dominator_tree.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "dominator_tree_test",
    srcs = ["dominator_tree_test.py"],
    deps = [
        ":dominator_tree",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
