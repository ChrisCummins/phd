# Labelled graphs for data dependence.

py_library(
    name = "data_dependence",
    srcs = ["data_dependence.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "data_dependence_test",
    srcs = ["data_dependence_test.py"],
    deps = [
        ":data_dependence",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:test",
    ],
)
