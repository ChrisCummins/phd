# Jasper is a small command line utility to run long-running MySQL queries.

py_binary(
    name = "jasper",
    srcs = ["jasper.py"],
    deps = [
        "//labm8:app",
        "//labm8:fs",
        "//labm8:text",
        "//tools/code_style/linters:linters_lib",
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
