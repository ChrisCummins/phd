# Scripts for training models.

py_binary(
    name = "devmap",
    srcs = ["devmap.py"],
    data = [
        "//deeplearning/ml4pl/models/ggnn",
        "//deeplearning/ml4pl/models/lstm:lstm_graph_classifier",
        "//deeplearning/ml4pl/models/zero_r",
    ],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
    ],
)
