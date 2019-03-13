# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "dash",
    srcs = ["dash.py"],
    deps = [
        requirement("dash"),
        requirement("plotly"),
        requirement("Click"),
        requirement("itsdangerous"),
        "//third_party/py/flask",
    ],
)
