load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "datastore",
    protos = ["datastore.proto"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
)

py_proto_library(
    name = "deepsmith",
    protos = ["deepsmith.proto"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    with_grpc = True,
)
