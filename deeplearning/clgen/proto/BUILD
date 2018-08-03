# Protocol buffers for Clgen.

load("@requirements//:requirements.bzl", "requirement")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

# Default to public visibility so that X_proto_library targets in other packages
# may depend on these protos using proto_deps. Specifying individual targets as
# public does not work since X_proto_library macros uses intermediate targets
# which must also be public.
package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "clgen_py_pb2",
    proto_deps = [
        ":corpus_py_pb2",
        ":model_py_pb2",
        ":sampler_py_pb2",
    ],
    protos = ["clgen.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "corpus_py_pb2",
    protos = ["corpus.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "internal_py_pb2",
    proto_deps = [
        ":corpus_py_pb2",
        ":model_py_pb2",
        ":sampler_py_pb2",
    ],
    protos = ["internal.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "model_py_pb2",
    proto_deps = [":corpus_py_pb2"],
    protos = ["model.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "sampler_py_pb2",
    protos = ["sampler.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "telemetry_py_pb2",
    protos = ["telemetry.proto"],
    deps = ["//third_party/py/protobuf"],
)
