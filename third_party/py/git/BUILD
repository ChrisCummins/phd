# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "git",
    srcs = ["git.py"],
    deps = [
        requirement("GitPython"),
        requirement("gitdb2"),
        requirement("smmap2"),
    ],
)
