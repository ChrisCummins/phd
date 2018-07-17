# Experiments in deep learning for source code loops.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "generate_loops",
    srcs = ["generate_loops.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
