# Test data and code for //phd.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

package(default_visibility = ["//phd:__subpackages__"])

cc_proto_library(
    name = "protos_cc",
    protos = ["protos.proto"],
    with_grpc = False,
)

filegroup(
    name = "data_file",
    testonly = 1,
    srcs = ["data_file.txt"],
    visibility = ["//phd:__subpackages__"],
)
