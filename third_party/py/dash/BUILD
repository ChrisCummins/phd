# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "dash",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("dash"),
        requirement("plotly"),
        "//third_party/py/flask",
        "//third_party/py/requests",
    ],
)
