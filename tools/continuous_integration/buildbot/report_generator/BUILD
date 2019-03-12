# Generate incremental test reports.

py_binary(
    name = "report_generator",
    srcs = ["report_generator.py"],
    visibility = ["//tools/continuous_integration/buildbot/report_generator:__subpackages__"],
    deps = [
        "//config:getconfig",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//tools/continuous_integration:bazel_test_db",
    ],
)
