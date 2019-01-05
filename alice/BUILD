# A little command executor.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "alice_py_pb2",
    protos = ["alice.proto"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)
