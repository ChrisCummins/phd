# This package demonstrates how to use protocol buffers in bazel.

# Load the python protobuf library rule.
load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

# Declare the proto here.
proto_library(
    name = "this_name_does_not_matter",
    srcs = [
        "another.proto",
        "example.proto",
    ],
)

# This rule instructs bazel to produce python protobuf files from a proto
# library. This is what python targets can depend on. The name of the rule is
# arbitrary - the path of the python files is derived from the proto file names.
python_proto_library(
    name = "also_this_name_does_not_matter",
    deps = [":this_name_does_not_matter"],
)

py_test(
    name = "py_proto_test",
    srcs = ["py_proto_test.py"],
    deps = [
        ":also_this_name_does_not_matter",
    ],
)
