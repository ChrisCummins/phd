# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "reorder_python_imports",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("reorder-python-imports"),
        "//third_party/py/aspy_refactor_imports",
        "//third_party/py/cached_property",
    ],
)
