py_library(
    name = "dominator_tree",
    srcs = ["dominator_tree.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_iterators",
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
