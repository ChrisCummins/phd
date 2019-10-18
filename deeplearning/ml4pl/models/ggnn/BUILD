py_library(
    name = "ggnn_base",
    data = ["//deeplearning/ncc/published_results:embeddings"],
    srcs = ["ggnn_base.py"],
    deps = [
        ":ggnn_utils",
        "//labm8:app",
        "//labm8:prof",
        "//labm8:system",
        "//labm8:jsonutil",
        "//labm8:bazelutil",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "ggnn_base_test",
    srcs = ["ggnn_base_test.py"],
    deps = [
        ":ggnn_base",
        "//third_party/py/pytest",
        "//labm8:test",
        "//labm8:app",
    ]
)

py_binary(
    name = "ggnn_graph_classifier",
    srcs = ["ggnn_graph_classifier.py"],
    deps = [
        ":ggnn_base",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "ggnn_node_classifier",
    srcs = ["ggnn_node_classifier.py"],
    deps = [
        ":ggnn_base",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "ggnn_utils",
    srcs = ["ggnn_utils.py"],
    deps = [
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
