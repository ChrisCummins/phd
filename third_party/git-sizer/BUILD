# git-sizer

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_binary(
    name = "git-sizer",
    srcs = ["git-sizer.sh"],
    data = select({
        "//:darwin": [
            "@git_sizer_mac//:git-sizer",
        ],
        "//conditions:default": [
            "@git_sizer_linux//:git-sizer",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "git-sizer_smoke_test",
    srcs = ["git-sizer_smoke_test.bats"],
    data = [
        ":git-sizer",
        "//labm8/sh:test",
    ],
)
