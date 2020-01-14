# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "aspy_refactor_imports",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("aspy.refactor-imports"),
    ],
)
