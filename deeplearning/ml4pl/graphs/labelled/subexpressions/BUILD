# Labelled graphs for global common subexpressions.

py_library(
    name = "subexpressions",
    srcs = ["subexpressions.py"],
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
    name = "subexpressions_test",
    srcs = ["subexpressions_test.py"],
    deps = [
        ":subexpressions",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)
