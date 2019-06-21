load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["parser.go"],
    importpath = "github.com/ChrisCummins/phd/compilers/toy/parser",
    visibility = ["//visibility:public"],
    deps = [
        "//compilers/toy/ast:go_default_library",
        "//compilers/toy/token:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["parser_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//compilers/toy/token:go_default_library",
        "@com_github_stretchr_testify//assert:go_default_library",
    ],
)
