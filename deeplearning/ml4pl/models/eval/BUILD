# Evaluating model performance.

py_binary(
    name = "export_leaderboard",
    srcs = ["export_leaderboard.py"],
    deps = [
        ":google_sheets",
        ":leaderboard",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:prof",
    ],
)

py_binary(
    name = "ault_export_leaderboard",
    srcs = ["ault_export_leaderboard.py"],
    deps = [
        ":google_sheets",
        ":leaderboard",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "google_sheets",
    srcs = ["google_sheets.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/gspread",
        "//third_party/py/oauth2client",
    ],
)

py_binary(
    name = "leaderboard",
    srcs = ["leaderboard.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:pdutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "recreate_graphs",
    srcs = ["recreate_graphs.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/models:log_analysis",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "test_leaderboard",
    srcs = ["test_leaderboard.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:pdutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)
