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
