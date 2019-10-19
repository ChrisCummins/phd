# This package defines modules for representing graphs as dictionaries.

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
