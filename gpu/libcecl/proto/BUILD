load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "libcecl_py_pb2",
    proto_deps = [
        "//gpu/clinfo/proto:clinfo_py_pb2",
    ],
    protos = ["libcecl.proto"],
    deps = ["//third_party/py/protobuf"],
)
