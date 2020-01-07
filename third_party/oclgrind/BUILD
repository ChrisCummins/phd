# Third party code.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_binary(
    name = "oclgrind",
    srcs = ["oclgrind.sh"],
    data = select({
        "//:darwin": ["@oclgrind_mac//:oclgrind"],
        "//conditions:default": ["@oclgrind_linux//:oclgrind"],
    }),
    visibility = ["//visibility:public"],
    deps = ["//labm8/sh:app"],
)

bats_test(
    name = "oclgrind_smoke_test",
    srcs = ["oclgrind_smoke_test.bats"],
    data = [
        ":oclgrind",
        "//labm8/sh:test",
    ],
)
