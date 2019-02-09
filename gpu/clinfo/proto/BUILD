# Protos for describing OpenCL devices.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility=["//visibility:public"])

cc_proto_library(
    name = "clinfo_cc_pb2",
    protos = ["clinfo.proto"],
)

py_proto_library(
    name = "clinfo_py_pb2",
    protos = ["clinfo.proto"],
)
