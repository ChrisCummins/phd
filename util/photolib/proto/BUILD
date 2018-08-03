# Protocol buffers used by //util/photolib.

load("@requirements//:requirements.bzl", "requirement")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "photolint_py_pb2",
    protos = ["photolint.proto"],
    visibility = ["//util/photolib:__subpackages__"],
    deps = ["//third_party/py/protobuf"],
)
