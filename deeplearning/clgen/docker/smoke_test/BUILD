# Smoke test of CLgen docker image.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

bats_test(
    name = "smoke_test",
    srcs = ["smoke_test.bats"],
    data = [
        "//deeplearning/clgen/docker:clgen.tar",
        "//deeplearning/clgen/tests/data/tiny",
        "//labm8/sh:test",
    ],
)
