# Machine learning models.

py_library(
    name = "base_utils",
    srcs = ["base_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "batch",
    srcs = ["batch.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
    ],
)

py_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    deps = [
        ":epoch",
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
    ],
)

py_library(
    name = "classifier_base",
    srcs = ["classifier_base.py"],
    data = ["//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:embeddings"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_utils",
        ":checkpoints",
        ":epoch",
        ":logger",
        "//:build_info",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "classifier_base_test",
    timeout = "long",
    srcs = ["classifier_base_test.py"],
    deps = [
        ":batch",
        ":classifier_base",
        ":epoch",
        ":log_database",
        ":logger",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "copy_database",
    srcs = ["copy_database.py"],
    deps = [
        ":log_database",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
    ],
)

py_library(
    name = "epoch",
    srcs = ["epoch.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        ":batch",
        "//labm8/py:app",
    ],
)

py_binary(
    name = "export_logs",
    srcs = ["export_logs.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        ":logger",
        ":schedules",
        "//labm8/py:app",
        "//labm8/py:google_sheets",
        "//labm8/py:pdutil",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "log_analysis",
    srcs = ["log_analysis.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scikit_learn",
    ],
)

py_test(
    name = "log_analysis_test",
    timeout = "long",
    srcs = ["log_analysis_test.py"],
    deps = [
        ":log_analysis",
        ":log_database",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_log_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "log_database",
    srcs = ["log_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch",
        ":checkpoints",
        ":epoch",
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:google_sheets",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:labdate",
        "//labm8/py:pdutil",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_database_test",
    timeout = "long",
    srcs = ["log_database_test.py"],
    deps = [
        ":log_database",
        "//deeplearning/ml4pl/testing:random_log_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        ":batch",
        ":checkpoints",
        ":epoch",
        ":log_database",
        ":schedules",
        "//:build_info",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//labm8/py:prof",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        ":classifier_base",
        ":epoch",
        ":logger",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//labm8/py:progress",
    ],
)

py_library(
    name = "schedules",
    srcs = ["schedules.py"],
    deps = [
        "//labm8/py:app",
    ],
)
