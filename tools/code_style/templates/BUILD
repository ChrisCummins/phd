# TODO: One line summary of package.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "proto",
    protos = ["proto.proto"],
)

py_library(
    name = "python",
    srcs = ["python.py"],
)

py_test(
    name = "python_test",
    size = "small",
    srcs = ["python_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
)
