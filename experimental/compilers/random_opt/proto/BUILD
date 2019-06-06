# Protocol buffers for random optimization.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "random_opt_pb",
    srcs = ["random_opt.proto"],
)

python_proto_library(
    name = "random_opt_pb_py",
    visibility = ["//visibility:public"],
    deps = [":random_opt_pb"],
)
