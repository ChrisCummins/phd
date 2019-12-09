# Scripts for migrating log databases.

py_binary(
    name = "prune_logs",
    srcs = ["prune_logs.py"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "prune_logs_test",
    srcs = ["prune_logs_test.py"],
    deps = [
        ":prune_logs",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
