# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "ascii_art",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("ascii_art"),
    ],
)
