# Protocol buffers for labm8.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "lockfile_py_pb2",
    protos = ["lockfile.proto"],
    visibility = ["//lib/labm8:__subpackages__"],
)

py_proto_library(
    name = "logging_py_pb2",
    protos = ["logging.proto"],
    visibility = ["//visibility:public"],
)

py_proto_library(
    name = "test_protos",
    testonly = 1,
    protos = ["test_protos.proto"],
    visibility = ["//lib/labm8:__subpackages__"],
)
