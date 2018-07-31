# Protocol buffers for DeepSmith.

load("@requirements//:requirements.bzl", "requirement")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "reachability_py_pb2",
    protos = ["reachability.proto"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = [requirement("protobuf")],
)
