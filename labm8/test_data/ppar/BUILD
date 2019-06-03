# Test data for //labm8:ppar_test.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")
load("@build_stack_rules_proto//cpp:cpp_proto_library.bzl", "cpp_proto_library")

package(default_visibility = ["//labm8:__subpackages__"])

proto_library(
    name = "protos",
    srcs = ["protos.proto"],
)

cpp_proto_library(
    name = "protos_pb_cc",
    deps = [":protos"],
)

python_proto_library(
    name = "protos_pb_py",
    deps = [":protos"],
)

cc_binary(
    name = "proto_worker",
    srcs = ["proto_worker.cc"],
    deps = [
        ":protos_pb_cc",
        "//phd:logging",
        "//phd:pbutil",
    ],
)

cc_binary(
    name = "proto_worker_requires_args",
    srcs = ["proto_worker_requires_args.cc"],
    deps = [
        "//phd:logging",
    ],
)

cc_binary(
    name = "unexpected_output_proto_worker",
    srcs = ["unexpected_output_proto_worker.cc"],
    deps = [
        ":protos_pb_cc",
        "//phd:logging",
    ],
)
