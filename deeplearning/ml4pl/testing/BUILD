# This package contains utility code for writing tests.

py_library(
    name = "testing_databases",
    testonly = 1,
    srcs = ["testing_databases.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
    ],
)

py_test(
    name = "testing_databases_test",
    srcs = ["testing_databases_test.py"],
    deps = [
        ":testing_databases",
        "//labm8/py:sqlutil",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)
