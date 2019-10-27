# Labelled graphs for reachability analysis.

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
