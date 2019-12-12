# Gated graph neural network.

py_binary(
    name = "ggnn",
    srcs = ["ggnn.py"],
    visibility = ["//deeplearning/ml4pl/experiments:__subpackages__"],
    deps = [
        ":ggnn_config",
        ":ggnn_modules",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:node_encoder",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:run",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/torch",
    ],
)

py_test(
    name = "ggnn_test",
    size = "enormous",
    timeout = "eternal",
    srcs = ["ggnn_test.py"],
    deps = [
        ":ggnn",
        ":ggnn_config",
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

py_library(
    name = "ggnn_config",
    srcs = ["ggnn_config.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_library(
    name = "ggnn_modules",
    srcs = ["ggnn_modules.py"],
    deps = [
        ":ggnn_config",
        "//labm8/py:app",
        "//third_party/py/torch",
    ],
)
