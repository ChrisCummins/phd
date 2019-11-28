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
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)

py_binary(
    name = "export_leaderboards",
    srcs = ["export_leaderboards.py"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/eval:google_sheets",
        "//deeplearning/ml4pl/models/eval:leaderboard",
        "//labm8/py:app",
        "//labm8/py:prof",
    ],
)
