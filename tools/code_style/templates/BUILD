# TODO: One line summary of package.

load("@requirements//:requirements.bzl", "requirement")
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
        "//third_party/py/absl",
    ],
)

py_test(
    name = "python_test",
    size = "small",
    srcs = ["python_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
