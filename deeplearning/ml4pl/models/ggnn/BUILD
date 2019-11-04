py_library(
    name = "ggnn_base",
    srcs = ["ggnn_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":ggnn_utils",
        "//deeplearning/ml4pl/models:classifier_base",
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
    visibility = ["//visibility:public"],
    deps = [
        ":ggnn_base",
        ":ggnn_utils",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "ggnn_node_classifier_smoke_test",
    size = "large",
    srcs = ["ggnn_node_classifier_smoke_test.py"],
    deps = [
        ":ggnn_node_classifier",
        "//deeplearning/ml4pl/models:classifier_smoke_tester",
        "//labm8:app",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ggnn_utils",
    srcs = ["ggnn_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
