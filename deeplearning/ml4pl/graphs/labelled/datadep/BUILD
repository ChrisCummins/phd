# Labelled graphs for data dependence.

py_library(
    name = "data_dependence",
    srcs = ["data_dependence.py"],
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
    name = "data_dependence_test",
    srcs = ["data_dependence_test.py"],
    deps = [
        ":data_dependence",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)
