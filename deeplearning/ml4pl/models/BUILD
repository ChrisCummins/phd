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
        "//labm8/py:ppar",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "classifier_base_test",
    srcs = ["classifier_base_test.py"],
    deps = [
        ":classifier_base",
        ":log_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "classifier_smoke_tester",
    testonly = 1,
    srcs = ["classifier_smoke_tester.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//third_party/py/numpy",
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

py_library(
    name = "log_analysis",
    srcs = ["log_analysis.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
        "//labm8/py:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scikit_learn",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_analysis_test",
    srcs = ["log_analysis_test.py"],
    deps = [
        ":log_analysis",
        ":log_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "log_database",
    srcs = ["log_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:pdutil",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_database_test",
    srcs = ["log_database_test.py"],
    deps = [
        ":log_database",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    deps = [
        ":batch",
        ":epoch",
        ":log_database",
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//third_party/py/numpy",
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
