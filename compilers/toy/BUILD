load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["compiler.go"],
    importpath = "github.com/ChrisCummins/phd/compilers/toy",
    visibility = ["//visibility:public"],
    deps = [
        "//compilers/toy/lexer:go_default_library",
        "//compilers/toy/parser:go_default_library",
    ],
)

go_binary(
    name = "toy",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "test_data",
    testonly = 1,
    srcs = glob([
        "test_data/**/*.c",
    ]) + ["test_data/test_compiler.sh"],
)

sh_binary(
    name = "compile",
    srcs = ["compile.sh"],
    data = [":toy"],
)

sh_test(
    name = "stage1_test",
    srcs = ["run_tests.sh"],
    args = ["1"],
    data = [
        ":compile",
        ":test_data",
    ],
)

sh_test(
    name = "stage2_test",
    srcs = ["run_tests.sh"],
    args = ["2"],
    data = [
        ":compile",
        ":test_data",
    ],
)

sh_test(
    name = "stage3_test",
    srcs = ["run_tests.sh"],
    args = ["3"],
    data = [
        ":compile",
        ":test_data",
    ],
)

#sh_test(
#    name = "stage4_test",
#    srcs = ["run_tests.sh"],
#    args = ["4"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
#
#sh_test(
#    name = "stage5_test",
#    srcs = ["run_tests.sh"],
#    args = ["5"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
#
#sh_test(
#    name = "stage6_test",
#    srcs = ["run_tests.sh"],
#    args = ["6"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
#
#sh_test(
#    name = "stage7_test",
#    srcs = ["run_tests.sh"],
#    args = ["7"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
#
#sh_test(
#    name = "stage8_test",
#    srcs = ["run_tests.sh"],
#    args = ["8"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
#
#sh_test(
#    name = "stage9_test",
#    srcs = ["run_tests.sh"],
#    args = ["9"],
#    data = [
#        ":compile",
#        ":test_data",
#    ],
#)
