# This package implements a native binary for tokenizing strings.

cc_library(
    name = "lexer",
    srcs = ["lexer.cc"],
    hdrs = ["lexer.h"],
    deps = [
        "//deeplearning/clgen/proto:internal_pb_cc",
        "//labm8/cpp:string",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "lexer_test",
    srcs = ["lexer_test.cc"],
    deps = [
        ":lexer",
        "//labm8/cpp:string",
        "//labm8/cpp:test",
    ],
)

cc_binary(
    name = "lexer_worker",
    srcs = ["lexer_worker.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":lexer",
        "//labm8/cpp:pbutil",
    ],
)
