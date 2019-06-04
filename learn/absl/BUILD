# Learning to use Google's abseil libraries.

load("@build_stack_rules_proto//cpp:cpp_proto_library.bzl", "cpp_proto_library")

cc_binary(
    name = "hello_absl",
    srcs = ["hello_absl.cc"],
    deps = [
        ":message_pb_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
    ],
)

proto_library(
    name = "message_pb",
    srcs = ["message.proto"],
)

cpp_proto_library(
    name = "message_pb_cc",
    deps = [":message_pb"],
)
