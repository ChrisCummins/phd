# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "typed_ast",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("typed-ast"),
    ],
)
