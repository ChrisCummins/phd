# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "progressbar",
    srcs = ["progressbar.py"],
    deps = [
        requirement("progressbar2"),
    ],
)
