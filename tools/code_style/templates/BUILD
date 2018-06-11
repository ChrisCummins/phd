# TODO: One line summary of package.

load("@requirements//:requirements.bzl", "requirement")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "proto_py_pb2",
    protos = ["proto.proto"],
    deps = [requirement("protobuf")],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    deps = [
        requirement("absl-py"),
    ],
)

py_test(
    name = "python_test",
    size = "small",
    srcs = ["python_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
