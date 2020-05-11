# Jasper is a small command line utility to run long-running MySQL queries.

load("@subpar//:subpar.bzl", "par_binary")
load("@cec_exports_repo//tools/bzl:exports_repo.bzl", "exports_repo")

exports_repo(
    name = "export",
    path_rename = {
        "util/jasper/README.md": "README.md",
        "util/jasper/LICENSE": "LICENSE",
    },
    remote = "git@github.com:ChrisCummins/jasper.git",
    tags = ["manual"],
    targets = [
        "//util/jasper/...",
    ],
)

par_binary(
    name = "jasper",
    srcs = ["jasper.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:text",
        "//third_party/py/sqlparse",
        "//util/lmk",
    ],
)

py_test(
    name = "jasper_test",
    srcs = ["jasper_test.py"],
    deps = [
        ":jasper",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [":jasper.par"],
)
