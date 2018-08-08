# Protocol buffers for benchmarks.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "benchmarks_py_pb2",
    protos = ["benchmarks.proto"],
    visibility = ["//datasets/benchmarks:__subpackages__"],
    deps = ["//third_party/py/protobuf"],
)
