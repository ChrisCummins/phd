# Zero-R models.

py_binary(
    name = "zero_r",
    srcs = ["zero_r.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:run",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "zero_r_test",
    srcs = ["zero_r_test.py"],
    deps = [
        ":zero_r",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)
