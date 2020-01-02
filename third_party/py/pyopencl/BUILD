# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "pyopencl",
    srcs = ["pyopencl.py"],
    deps = [
        requirement("pyopencl"),
        requirement("pytools"),  # Implicit dependency.
    ],
)
