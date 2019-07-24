# Jasper is a small command line utility to run long-running MySQL queries.

load("@subpar//:subpar.bzl", "par_binary")

py_binary(
    name = "jasper",
    srcs = ["jasper.py"],
    deps = [
        "//labm8:app",
        "//labm8:fs",
        "//labm8:text",
        "//tools/code_style/formatters:sql_formatter",
        "//util/lmk",
    ],
)

py_test(
    name = "jasper_test",
    srcs = ["jasper_test.py"],
    deps = [
        ":jasper",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

par_binary(
    name = "jasper_bin",
    srcs = ["jasper.py"],
    main = "jasper.py",
    deps = [":jasper"],
)
