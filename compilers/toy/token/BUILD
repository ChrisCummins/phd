load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "token.go",
        "token_stream.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy/token",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "token_stream_test.go",
        "token_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["@com_github_stretchr_testify//assert:go_default_library"],
)
