# Protocol buffers used by //util/photolib.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "photolint_pb",
    srcs = ["photolint.proto"],
)

python_proto_library(
    name = "photolint_pb_py",
    visibility = ["//util/photolib:__subpackages__"],
    deps = [":photolint_pb"],
)
