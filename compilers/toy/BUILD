# A toy compiler written in go.

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_binary(
    name = "compiler",
    srcs = [
        "compiler.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy",
)

go_library(
    name = "lexer",
    srcs = [
        "lexer.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy",
)

go_test(
    name = "lexer_test",
    srcs = [
        "lexer.go",
        "lexer_test.go",
    ],
    importpath = "github.com/ChrisCummins/phd/compilers/toy",
)
