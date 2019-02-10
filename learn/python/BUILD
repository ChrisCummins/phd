# Learning Python.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_test(
    name = "ascii_chart_test",
    srcs = ["ascii_chart_test.py"],
    deps = [
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/ascii_art",
    ],
)

cc_proto_library(
    name = "cpp_interop_proto_cc",
    protos = ["cpp_interop.proto"],
)

py_proto_library(
    name = "cpp_interop_proto_py_pb2",
    protos = ["cpp_interop.proto"],
)

cc_binary(
    name = "cpp_interop_bin",
    srcs = [
        "cpp_interop_bin.cc",
    ],
    deps = [
        ":cpp_interop_proto_cc",
        "//phd:macros",
        "//phd:pbutil",
    ],
)

cc_binary(
    name = "cpp_interop_inplace_bin",
    srcs = [
        "cpp_interop_inplace_bin.cc",
    ],
    deps = [
        ":cpp_interop_proto_cc",
        "//phd:macros",
        "//phd:pbutil",
    ],
)

py_test(
    name = "cpp_interop_test",
    srcs = ["cpp_interop_test.py"],
    data = [
        ":cpp_interop_bin",
        ":cpp_interop_inplace_bin",
    ],
    deps = [
        ":cpp_interop_proto_py_pb2",
        "//labm8:bazelutil",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "debugger",
    srcs = ["debugger.py"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/ipdb",
    ],
)
