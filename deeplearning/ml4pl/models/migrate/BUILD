# Scripts for migrating log databases.

py_binary(
    name = "copy_logs",
    srcs = ["copy_logs.py"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "copy_logs_test",
    srcs = ["copy_logs_test.py"],
    deps = [
        ":copy_logs",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "prune_logs",
    srcs = ["prune_logs.py"],
    deps = [
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
    ],
)

py_test(
    name = "prune_logs_test",
    srcs = ["prune_logs_test.py"],
    deps = [
        ":prune_logs",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)
