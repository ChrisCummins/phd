# Root package for code that works with GitHub.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "github_py_pb2",
    protos = ["github.proto"],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":github_py_pb2",
        "//labm8:app",
        "//third_party/py/github",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)
