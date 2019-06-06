# Protobufs for project fish.
load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "fish_pb",
    srcs = ["fish.proto"],
)

python_proto_library(
    name = "fish_pb_py",
    visibility = ["//visibility:public"],
    deps = [":fish_pb"],
)
