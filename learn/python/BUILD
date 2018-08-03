# Learning Python.

load("@requirements//:requirements.bzl", "requirement")

py_test(
    name = "ascii_chart",
    srcs = ["ascii_chart.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("absl-py"),
        requirement("ascii_art"),
        requirement("pytest"),
    ],
)
