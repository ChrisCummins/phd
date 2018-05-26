# The specification of the global configuration.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "config",
    protos = ["config.proto"],
    visibility = ["//visibility:public"],
    deps = ["//lib/labm8:crypto"],
)
