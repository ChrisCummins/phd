# A wrapper around lru_dict pip package to pull in undeclared dependencies.
#
# If a python target requires lru_dict, it should depend on this package
# (i.e. //third_party/py/lru_dict), instead of requirement("lru_dict").

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "lru_dict",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("lru-dict"),
    ],
)
