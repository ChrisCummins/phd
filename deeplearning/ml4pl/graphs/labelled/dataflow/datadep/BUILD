# Labelled graphs for data dependence.

py_library(
    name = "data_dependence",
    srcs = ["data_dependence.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "data_dependence_test",
    srcs = ["data_dependence_test.py"],
    deps = [
        ":data_dependence",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
