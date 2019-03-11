# TODO: One line summary of package.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "proto_py_pb2",
    protos = ["proto.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    deps = [
        "//labm8:app",
    ],
)

py_test(
    name = "python_test",
    size = "small",
    srcs = ["python_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)
