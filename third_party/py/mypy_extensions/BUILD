# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "mypy_extensions",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("mypy-extensions"),
    ],
)
