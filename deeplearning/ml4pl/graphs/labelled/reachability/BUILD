# Labelled graphs for reachability analysis.

py_library(
    name = "reachability",
    srcs = ["reachability.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_query",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "reachability_test",
    srcs = ["reachability_test.py"],
    deps = [
        ":reachability",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
