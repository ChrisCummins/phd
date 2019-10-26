# This package defines modules for representing graphs as dictionaries.

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

py_library(
    name = "graph_dict",
    srcs = ["graph_dict.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_dict_test",
    srcs = ["graph_dict_test.py"],
    deps = [
        ":graph_dict",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
