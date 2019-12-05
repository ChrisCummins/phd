py_binary(
    name = "ggnn",
    srcs = ["ggnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":ggnn_utils",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/models:base_utils",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

# TODO(github.com/ChrisCummins/ProGraML/issues/24): Totally broken.
# py_test(
#     name = "ggnn_test",
#     srcs = ["ggnn_test.py"],
#     deps = [
#         ":ggnn",
#         "//deeplearning/ml4pl/models:log_database",
#         "//labm8/py:app",
#         "//labm8/py:test",
#     ],
# )
#
# py_test(
#     name = "ggnn_smoke_test",
#     size = "large",
#     srcs = ["ggnn_smoke_test.py"],
#     deps = [
#         ":ggnn",
#         "//deeplearning/ml4pl/models:classifier_smoke_tester",
#         "//labm8/py:app",
#         "//labm8/py:test",
#         "//third_party/py/numpy",
#     ],
# )

py_library(
    name = "ggnn_utils",
    srcs = ["ggnn_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
