# Experiments in deep learning for source code loops.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "generate_loops",
    srcs = ["generate_loops.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
