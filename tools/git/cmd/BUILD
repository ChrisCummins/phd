# Command-line tools for git.

load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "export_subtree",
    srcs = ["export_subtree.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/git",
        "//tools/git:export_subtree",
    ],
)
