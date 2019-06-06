# Protos for describing 'dpack' data packages.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "dpack_pb",
    srcs = ["dpack.proto"],
)

python_proto_library(
    name = "dpack_pb_py",
    visibility = ["//system/dpack:__subpackages__"],
    deps = [":dpack_pb"],
)
