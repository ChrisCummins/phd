# A module for monitoring memory usage of a python program
# https://pypi.org/project/memory-profiler/

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "memory_profiler",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("memory-profiler"),
        "//third_party/py/psutil",
    ],
)
