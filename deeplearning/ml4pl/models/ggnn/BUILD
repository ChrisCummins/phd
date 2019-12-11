# Gated graph neural network.

py_binary(
    name = "ggnn",
    srcs = ["ggnn.py"],
    deps = [
        ":ggnn_config",
        ":ggnn_modules",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:node_encoder",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
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

py_library(
    name = "ggnn_config",
    srcs = ["ggnn_config.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database_stats",
        "//labm8/py:app",
    ],
)

py_library(
    name = "ggnn_modules",
    srcs = ["ggnn_modules.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/torch",
    ],
)
