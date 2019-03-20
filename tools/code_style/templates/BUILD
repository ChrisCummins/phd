# TODO: One line summary of package.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "proto_py_pb2",
    protos = ["proto.proto"],
    deps = ["//third_party/py/protobuf"],
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
