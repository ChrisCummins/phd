# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "checksumdir",
    srcs = ["checksumdir.py"],
    deps = [
        requirement("checksumdir"),
    ],
)
