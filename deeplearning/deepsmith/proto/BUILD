load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "datastore",
    protos = ["datastore.proto"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
)

py_proto_library(
    name = "deepsmith",
    protos = ["deepsmith.proto"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    with_grpc = True,
)

py_proto_library(
    name = "test_protos",
    testonly = 1,
    protos = ["test_protos.proto"],
)

py_library(
    name = "pbutil",
    srcs = ["pbutil.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
)

py_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.py"],
    deps = [
        ":pbutil",
        ":test_protos",
    ],
)
