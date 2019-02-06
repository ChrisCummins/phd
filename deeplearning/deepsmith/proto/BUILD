# Protocol buffers for DeepSmith.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "datastore_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
    ],
    protos = ["datastore.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "deepsmith_py_pb2",
    protos = ["deepsmith.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "generator_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
        "//deeplearning/clgen/proto:clgen_py_pb2",
    ],
    protos = ["generator.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "harness_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
    ],
    protos = ["harness.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "service_py_pb2",
    protos = ["service.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)
