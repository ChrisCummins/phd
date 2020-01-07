# Smoke test for cldrive.
load("@rules_bats//:bats.bzl", "bats_test")

bats_test(
    name = "smoke_test",
    srcs = ["smoke_test.bats"],
    data = [
        "//gpu/cldrive",
        "//gpu/oclgrind",
        "//labm8/sh:test",
    ],
)
