# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GPL v2

py_library(
    name = "inotify",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("inotify"),
    ],
)
