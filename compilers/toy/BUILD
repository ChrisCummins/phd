load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["compiler.go"],
    importpath = "github.com/ChrisCummins/phd/compilers/toy",
    visibility = ["//visibility:public"],
    deps = [
        "//compilers/toy/lexer:go_default_library",
        "//compilers/toy/token:go_default_library",
    ],
)

go_binary(
    name = "toy",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
