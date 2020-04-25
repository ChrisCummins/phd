py_library(
    name = "graph_util",
    srcs = ["graph_util.py"],
    visibility = ["//learn/daily/d181212_graph_nets_shortest_path:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:labtypes",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "graph_util_test",
    srcs = ["graph_util_test.py"],
    deps = [
        ":graph_util",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//learn/daily/d181212_graph_nets_shortest_path:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/graph_nets",
        "//third_party/py/networkx",
        "//third_party/py/tensorflow",
    ],
)
