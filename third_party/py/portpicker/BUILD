# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "portpicker",
    srcs = ["portpicker.py"],
    deps = [
        requirement("portpicker"),
    ],
)
