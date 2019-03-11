# Generate incremental test reports.

py_library(
    name = "bazel_test_db",
    srcs = ["bazel_test_db.py"],
    deps = [
        "//labm8:app",
        "//labm8:prof",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "report_generator",
    srcs = ["report_generator.py"],
    visibility = ["//tools/continuous_integration/buildbot/report_generator:__subpackages__"],
    deps = [
        ":bazel_test_db",
        "//config:getconfig",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
    ],
)
