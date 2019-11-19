# Machine learning models.

py_library(
    name = "base_utils",
    srcs = ["base_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "classifier_base",
    srcs = ["classifier_base.py"],
    data = ["//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:embeddings"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_utils",
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
        "//labm8:app",
        "//labm8:test",
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
        "//labm8:app",
        "//third_party/py/numpy",
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
