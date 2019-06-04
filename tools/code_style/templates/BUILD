# TODO: One line summary of package.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "proto_pb",
    srcs = ["proto.proto"],
)

python_proto_library(
    name = "proto_pb_py",
    deps = [":proto_pb"],
)

py_binary(
    name = "py_binary",
    srcs = ["py_binary.py"],
    deps = [
        "//labm8:app",
    ],
)

py_library(
    name = "py_library",
    srcs = ["py_library.py"],
    deps = [
        "//labm8:app",
    ],
)

py_test(
    name = "py_test",
    size = "small",
    srcs = ["py_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)
