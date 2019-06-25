load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "expression.go",
        "function.go",
        "literal.go",
        "program.go",
        "return.go",
        "unary_op.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy/ast",
    visibility = ["//visibility:public"],
    deps = ["//compilers/toy/token:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "codegen_test.go",
        "string_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["@com_github_stretchr_testify//assert:go_default_library"],
)
