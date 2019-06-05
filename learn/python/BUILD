# Learning Python.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_test(
    name = "ascii_chart_test",
    srcs = ["ascii_chart_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/ascii_art",
    ],
)

proto_library(
    name = "cpp_interop_pb",
    srcs = ["cpp_interop.proto"],
)

cc_proto_library(
    name = "cpp_interop_pb_cc",
    deps = [":cpp_interop_pb"],
)

python_proto_library(
    name = "cpp_interop_proto_pb_py",
    deps = [":cpp_interop_pb"],
)

cc_binary(
    name = "cpp_interop_bin",
    srcs = [
        "cpp_interop_bin.cc",
    ],
    deps = [
        ":cpp_interop_pb_cc",
        "//phd:logging",
        "//phd:pbutil",
    ],
)

cc_binary(
    name = "cpp_interop_inplace_bin",
    srcs = [
        "cpp_interop_inplace_bin.cc",
    ],
    deps = [
        ":cpp_interop_pb_cc",
        "//phd:logging",
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
        ":cpp_interop_proto_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "debugger",
    srcs = ["debugger.py"],
    deps = [
        "//labm8:app",
        "//third_party/py/ipdb",
    ],
)
