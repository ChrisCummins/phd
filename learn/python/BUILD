# Learning Python.

load("@requirements//:requirements.bzl", "requirement")

py_test(
    name = "ascii_chart",
    srcs = ["ascii_chart.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl",
        requirement("ascii_art"),
        "//third_party/py/pytest",
    ],
)
