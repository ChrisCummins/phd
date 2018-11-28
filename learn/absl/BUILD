# Learning to use Google's abseil libraries.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

cc_binary(
    name = "hello_absl",
    srcs = ["hello_absl.cc"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/container:flat_hash_map",
        ":message",
    ]
)

cc_proto_library(
  name = "message",
  protos = ["message.proto"],
)
