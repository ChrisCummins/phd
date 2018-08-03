load("@requirements//:requirements.bzl", "requirement")

py_test(
    name = "batch_linear_regression",
    size = "small",
    srcs = ["batch_linear_regression.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl",
        requirement("six"),
    ],
)
