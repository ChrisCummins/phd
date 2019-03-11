# Differential tests for DeepSmith.

py_library(
    name = "difftests",
    srcs = ["difftests.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
    ],
)

py_test(
    name = "difftests_test",
    srcs = ["difftests_test.py"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//gpu/cldrive/legacy:args",
        "//labm8:app",
    ],
)
