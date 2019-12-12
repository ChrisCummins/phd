# Labelled graphs for live-out variables.

py_library(
    name = "liveness",
    srcs = ["liveness.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "liveness_test",
    size = "large",
    srcs = ["liveness_test.py"],
    deps = [
        ":liveness",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
