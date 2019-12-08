# This package contains utility code for writing tests.

py_library(
    name = "random_graph_tuple_generator",
    testonly = 1,
    srcs = ["random_graph_tuple_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_networkx_generator",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//labm8/py:test",
    ],
)

py_test(
    name = "random_graph_tuple_generator_test",
    timeout = "long",
    srcs = ["random_graph_tuple_generator_test.py"],
    deps = [
        ":random_graph_tuple_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_graph_tuple_database_generator",
    testonly = 1,
    srcs = ["random_graph_tuple_database_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_graph_tuple_generator",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
    ],
)

py_test(
    name = "random_graph_tuple_database_generator_test",
    timeout = "long",
    srcs = ["random_graph_tuple_database_generator_test.py"],
    deps = [
        ":random_graph_tuple_database_generator",
        ":testing_databases",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_log_database_generator",
    testonly = 1,
    srcs = ["random_log_database_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_graph_tuple_database_generator",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
    ],
)

py_test(
    name = "random_log_database_generator_test",
    timeout = "long",
    srcs = ["random_log_database_generator_test.py"],
    deps = [
        ":random_log_database_generator",
        ":testing_databases",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_library(
    name = "random_networkx_generator",
    testonly = 1,
    srcs = ["random_networkx_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_programl_generator",
        "//deeplearning/ml4pl/graphs:programl",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "random_networkx_generator_test",
    timeout = "long",
    srcs = ["random_networkx_generator_test.py"],
    deps = [
        ":random_graph_tuple_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_programl_generator",
    testonly = 1,
    srcs = ["random_programl_generator.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
    ],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)

py_test(
    name = "random_programl_generator_test",
    srcs = ["random_programl_generator_test.py"],
    deps = [
        ":random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_library(
    name = "testing_databases",
    testonly = 1,
    srcs = ["testing_databases.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
    ],
)

py_test(
    name = "testing_databases_test",
    srcs = ["testing_databases_test.py"],
    deps = [
        ":testing_databases",
        "//labm8/py:sqlutil",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)
