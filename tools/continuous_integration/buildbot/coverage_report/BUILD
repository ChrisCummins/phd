# Test coverage rreports.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "coverage_report",
    srcs = ["coverage_report.py"],
    deps = [
        "//labm8:app",
    ],
)

py3_image(
    name = "image",
    srcs = ["coverage_report.py"],
    base = "@base//image",
    main = "coverage_report.py",
    deps = [
        ":coverage_report",
    ],
)
