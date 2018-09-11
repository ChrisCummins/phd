# The FreeFous task manager.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_binary(
    name = "freefocus",
    srcs = ["freefocus.py"],
)

py_proto_library(
    name = "freefocus_py_pb2",
    srcs = ["freefocus.proto"],
    deps = ["//third_party/py/protobuf"],
    visibility = ["//visibility:public"],
)
