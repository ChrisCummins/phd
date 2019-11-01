# Machine learning models.

py_library(
    name = "classifier_base",
    srcs = ["classifier_base.py"],
    data = ["//deeplearning/ncc/published_results:embeddings"],
    visibility = ["//visibility:public"],
    deps = [
        "//:build_info",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:decorators",
        "//labm8:humanize",
        "//labm8:jsonutil",
        "//labm8:ppar",
        "//labm8:prof",
        "//labm8:system",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
    ],
)

py_library(
    name = "log_analysis",
    srcs = ["log_analysis.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//labm8:app",
        "//labm8:decorators",
        "//labm8:humanize",
        "//labm8:labtypes",
        "//labm8:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_analysis_test",
    srcs = ["log_analysis_test.py"],
    deps = [
        ":log_analysis",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "log_database",
    srcs = ["log_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:labdate",
        "//labm8:pdutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_database_test",
    srcs = ["log_database_test.py"],
    deps = [
        ":log_database",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "log_writer",
    srcs = ["log_writer.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//labm8:app",
        "//labm8:jsonutil",
    ],
)

py_test(
    name = "log_writer_test",
    srcs = ["log_writer_test.py"],
    deps = [
        ":log_writer",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "batch_logger",
    srcs = ["batch_logger.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//labm8:app",
    ],
)

py_test(
    name = "batch_logger_test",
    srcs = ["batch_logger_test.py"],
    deps = [
        ":batch_logger",
        "//labm8:app",
        "//labm8:test",
    ],
)
