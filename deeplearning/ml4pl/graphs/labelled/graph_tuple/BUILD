# This package defines modules for representing graphs as dictionaries.

py_library(
    name = "graph_batcher",
    srcs = ["graph_batcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs:graph_database_reader",
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_tuple_test",
    srcs = ["graph_tuple_test.py"],
    deps = [
        ":graph_tuple",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
