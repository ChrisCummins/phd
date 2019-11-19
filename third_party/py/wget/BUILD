# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # Public domain

py_library(
    name = "wget",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("wget"),
    ],
)
