# Intermediate representations.

py_binary(
    name = "ir_database",
    srcs = ["ir_database.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "ir_database_test",
    srcs = ["ir_database_test.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":ir_database",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    deps = [
        ":ir_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "splitters_test",
    srcs = ["splitters_test.py"],
    deps = [
        ":ir_database",
        ":splitters",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
