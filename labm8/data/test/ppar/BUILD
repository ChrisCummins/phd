# Test data for //labm8:ppar_test.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility = ["//labm8:__subpackages__"])

cc_proto_library(
    name = "protos_cc",
    protos = ["protos.proto"],
)

py_proto_library(
    name = "protos_py_pb2",
    protos = ["protos.proto"],
)

cc_binary(
    name = "proto_worker",
    srcs = ["proto_worker.cc"],
    deps = [
        ":protos_cc",
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
        ":protos_cc",
        "//phd:logging",
    ],
)
