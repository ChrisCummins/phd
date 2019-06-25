load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["gorm_example.go"],
    importpath = "github.com/ChrisCummins/phd/learn/go/gorm_example",
    visibility = ["//visibility:private"],
    deps = [
        "@com_github_jinzhu_gorm//:go_default_library",
        "@com_github_jinzhu_gorm//dialects/sqlite:go_default_library",
    ],
)

go_binary(
    name = "gorm_example",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
