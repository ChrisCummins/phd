# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GPLv2

py_library(
    name = "pyfiglet",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("pyfiglet"),
    ],
)
