# Labelled graphs for global common subexpressions.

py_library(
    name = "subexpressions",
    srcs = ["subexpressions.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_iterators",
        "//deeplearning/ml4pl/graphs:graph_query",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "subexpressions_test",
    srcs = ["subexpressions_test.py"],
    deps = [
        ":subexpressions",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
