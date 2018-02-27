load("@protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "dsmith",
    srcs = ["dsmith.proto"],
)

py_proto_library(
    name = "dsmith_py",
    protoc = "@com_google_protobuf//:protoc",
    deps = [":dsmith"],
)
