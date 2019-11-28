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
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_batcher_test",
    srcs = ["graph_batcher_test.py"],
    deps = [
        ":graph_batcher",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_tuple_test",
    srcs = ["graph_tuple_test.py"],
    deps = [
        ":graph_tuple",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
