load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "cldrive_cc_pb2",
    proto_deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//gpu/libcecl/proto:libcecl_cc_pb2",
    ],
    protos = ["cldrive.proto"],
)

py_proto_library(
    name = "cldrive_py_pb2",
    proto_deps = [
        "//gpu/clinfo/proto:clinfo_py_pb2",
        "//gpu/libcecl/proto:libcecl_py_pb2",
    ],
    protos = ["cldrive.proto"],
)
