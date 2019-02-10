# The specification of the global configuration.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "config",
    protos = ["config.proto"],
    deps = ["//labm8:crypto"],
)
