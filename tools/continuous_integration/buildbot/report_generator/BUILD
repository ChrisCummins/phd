# Generate incremental test reports.

py_library(
    name = "bazel_test_db",
    srcs = ["bazel_test_db.py"],
    deps = [
        "//labm8:prof",
        "//labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "report_generator",
    srcs = ["report_generator.py"],
    deps = [
        ":bazel_test_db",
        "//config:getconfig",
        "//third_party/py/absl",
    ],
)
