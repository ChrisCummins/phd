load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

cc_binary(
    name = "lexer",
)

go_library(
    name = "go_default_library",
    srcs = [
        "lexer.go",
        "state_function.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy/lexer",
    visibility = ["//visibility:public"],
    deps = ["//compilers/toy/token:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["lexer_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//compilers/toy/token:go_default_library",
        "@com_github_stretchr_testify//assert:go_default_library",
    ],
)
