# Protocol buffers for DeepSmith.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

# The set of protocol buffers used by deepsmith. Note that proto files which
# include other protos must be contained in the same py_proto_library target.
py_proto_library(
    name = "deepsmith",
    protos = [
        "datastore.proto",
        "deepsmith.proto",
        "generator.proto",
        "harness.proto",
        "service.proto",
    ],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    with_grpc = True,
)
