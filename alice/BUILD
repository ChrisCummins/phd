# A little command executor.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "alice_py_pb2",
    proto_deps = [
        "//config/proto:config",
    ],
    protos = ["alice.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_library(
    name = "bazel",
    srcs = ["bazel.py"],
    visibility = ["//alice:__subpackages__"],
    deps = [
        "//labm8:fs",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "bazel_test",
    srcs = ["bazel_test.py"],
    data = ["//alice/test:dummy_target"],
    deps = [
        ":bazel",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "git_repo",
    srcs = ["git_repo.py"],
    visibility = ["//alice:__subpackages__"],
    deps = [
        ":alice_py_pb2",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "git_repo_test",
    srcs = ["git_repo_test.py"],
    deps = [
        ":git_repo",
        "//config:getconfig",
        "//labm8:pbutil",
        "//labm8:test",
        "//third_party/py/git",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [
        ":alice_py_pb2",
        ":git_repo",
        "//third_party/py/absl",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    deps = [
        ":run",
        "//labm8:test",
    ],
)

py_library(
    name = "xdata",
    srcs = ["xdata.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "xdata_test",
    srcs = ["xdata_test.py"],
    deps = [
        ":xdata",
        "//labm8:test",
    ],
)
