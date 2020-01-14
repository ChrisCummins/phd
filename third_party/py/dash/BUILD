# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "dash",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("dash"),
        requirement("dash-core-components"),
        requirement("dash-html-components"),
        requirement("dash-renderer"),
        requirement("dash-table"),
        requirement("plotly"),
        "//third_party/py/click",
        "//third_party/py/decorator",
        "//third_party/py/flask",
        "//third_party/py/requests",
    ],
)
