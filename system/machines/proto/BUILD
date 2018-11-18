# Protos for managing machines.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "data_tiers_pb2_py",
    protos = ["data_tiers.proto"],
    visibility = ["//system/machines:__subpackages__"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "machine_spec_pb2_py",
    protos = ["machine_spec.proto"],
    visibility = ["//system/machines:__subpackages__"],
    deps = ["//third_party/py/protobuf"],
)
