# Learning Python.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_test(
    name = "ascii_chart",
    srcs = ["ascii_chart.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl",
        "//third_party/py/ascii_art",
        "//third_party/py/pytest",
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
