# Generate incremental test reports.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

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

py3_image(
    name = "image",
    srcs = ["report_generator.py"],
    base = "@base//image",
    main = "report_generator.py",
    deps = [
        ":report_generator",
    ],
)
