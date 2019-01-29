# Models for learning over compiler CFGs.

py_binary(
    name = "graph_model",
    srcs = ["graph_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:labdate",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/graph_nets",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "graph_model_test",
    srcs = ["graph_model_test.py"],
    deps = [
        ":graph_model",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)
