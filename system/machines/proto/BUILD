# Protos for managing machines.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "data_tiers",
    srcs = ["data_tiers.proto"],
)

python_proto_library(
    name = "data_tiers_pb_py",
    visibility = ["//system/machines:__subpackages__"],
    deps = [":data_tiers"],
)

proto_library(
    name = "machine_spec_pb",
    srcs = ["machine_spec.proto"],
)

python_proto_library(
    name = "machine_spec_pb_py",
    visibility = ["//system/machines:__subpackages__"],
    deps = [":machine_spec_pb"],
)
