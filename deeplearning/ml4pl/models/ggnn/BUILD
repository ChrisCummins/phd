py_library(
    name = "ggnn_base",
    srcs = ["ggnn_base.py"],
    data = ["//deeplearning/ncc/published_results:embeddings"],
    deps = [
        ":ggnn_utils",
        ":graph_batcher",
        "//:build_info",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//labm8:jsonutil",
        "//labm8:prof",
        "//labm8:system",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "ggnn_base_test",
    srcs = ["ggnn_base_test.py"],
    deps = [
        ":ggnn_base",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "ggnn_graph_classifier",
    srcs = ["ggnn_graph_classifier.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":ggnn_base",
        ":ggnn_utils",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
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
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "ggnn_utils",
    srcs = ["ggnn_utils.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "graph_batcher",
    srcs = ["graph_batcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs:graph_database_reader",
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//deeplearning/ml4pl/graphs/labelled/graph_dict",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)
