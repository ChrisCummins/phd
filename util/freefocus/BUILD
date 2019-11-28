# The FreeFous task manager.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_binary(
    name = "freefocus",
    srcs = ["freefocus.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":sql",
        "//labm8/py:app",
    ],
)

proto_library(
    name = "freefocus_pb",
    srcs = ["freefocus.proto"],
)

python_proto_library(
    name = "freefocus_pb_py",
    visibility = ["//visibility:public"],
    deps = [":freefocus_pb"],
)

py_library(
    name = "sql",
    srcs = ["sql.py"],
    visibility = ["//util/freefocus:__subpackages__"],
    deps = [
        ":freefocus_pb_py",
        "//labm8/py:app",
        "//labm8/py:labdate",
        "//labm8/py:sqlutil",
    ],
)

py_test(
    name = "sql_test",
    srcs = ["sql_test.py"],
    deps = [
        ":freefocus_pb_py",
        ":sql",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "sql_ops",
    srcs = ["sql_ops.py"],
    visibility = ["//util/freefocus:__subpackages__"],
)
