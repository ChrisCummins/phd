# Smoke test for cldrive.
load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

bats_test(
    name = "smoke_test",
    srcs = ["smoke_test.bats"],
    data = [
        "//gpu/cldrive",
        "//gpu/oclgrind",
        "//labm8/sh:test",
    ],
)
