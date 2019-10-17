# Gated graph neural network experiments.

py_library(
    name = "graph_database",
    srcs = ["graph_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ]
)

py_test(
    name = "graph_database_test",
    srcs = ["graph_database_test.py"],
    deps = [
        "//labm8:test",
        "//labm8:app",
        "//third_party/py/networkx",
        ":graph_database",
    ]
)

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        "//experimental/compilers/reachability:control_and_data_flow_graph",
        "//experimental/compilers/reachability:database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "make_reachability_dataset_test",
    srcs = ["make_reachability_dataset_test.py"],
    deps = [
        "//labm8:test",
        "//labm8:app",
        "//third_party/py/networkx",
        ":make_reachability_dataset",
    ]
)
