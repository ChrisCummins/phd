# Unlabelled program graphs.

# TODO(github.com/ChrisCummins/ProGraML/issues/28): Fix me.
#py_binary(
#    name = "make_unlabelled_graph_dataset",
#    srcs = ["make_unlabelled_graph_dataset.py"],
#    visibility = ["//visibility:public"],
#    deps = [
#        "//deeplearning/ml4pl/ir:ir_database",
#        "//deeplearning/ml4pl/graphs:database_exporters",
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
#        "//labm8/py:app",
#        "//labm8/py:prof",
#    ],
#)

py_binary(
    name = "split",
    srcs = ["split.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/ir:split",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "split_test",
    srcs = ["split_test.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled:split",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/ir:split",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "unlabelled_graph_database",
    srcs = ["unlabelled_graph_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:labdate",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "unlabelled_graph_database_test",
    srcs = ["unlabelled_graph_database_test.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)
