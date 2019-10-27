# Labelled graphs for live-out variables.

py_library(
    name = "liveness",
    srcs = ["liveness.py"],
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
    name = "liveness_test",
    srcs = ["liveness_test.py"],
    deps = [
        ":liveness",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)
