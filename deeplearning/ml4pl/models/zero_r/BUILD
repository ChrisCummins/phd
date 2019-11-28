# Zero-R models.

py_binary(
    name = "zero_r",
    srcs = ["zero_r.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "zero_r_smoke_test",
    srcs = ["zero_r_smoke_test.py"],
    deps = [
        ":zero_r",
        "//deeplearning/ml4pl/models:classifier_smoke_tester",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)
