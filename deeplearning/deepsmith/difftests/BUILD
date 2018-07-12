# Differential tests for DeepSmith.

load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "difftests",
    srcs = ["difftests.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
    ],
)

py_test(
    name = "difftests_test",
    srcs = ["difftests_test.py"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//gpu/cldrive",
        requirement("absl-py"),
    ],
)
