# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "ascii_art",
    srcs = ["ascii_art.py"],
    deps = [
        requirement("ascii_art"),
    ],
)
