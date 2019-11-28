# Continuous integration testing for this project.

py_library(
    name = "bazel_test_db",
    srcs = ["bazel_test_db.py"],
    visibility = ["//tools/continuous_integration:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)
