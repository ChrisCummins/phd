# Protobufs for project fish.
load("@requirements//:requirements.bzl", "requirement")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

# Default to public visibility so that X_proto_library targets in other packages
# may depend on these protos using proto_deps. Specifying individual targets as
# public does not work since X_proto_library macros uses intermediate targets
# which must also be public.
package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "fish_py_pb2",
    protos = ["fish.proto"],
    deps = [requirement("protobuf")],
)
