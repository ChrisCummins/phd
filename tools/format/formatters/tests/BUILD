py_library(
    name = "testing",
    testonly = 1,
    srcs = ["testing.py"],
    deps = [
        "//tools/format:formatter",
    ],
)

py_test(
    name = "text_test",
    srcs = ["text_test.py"],
    deps = [
        ":testing",
        "//labm8/py:fs",
        "//labm8/py:test",
        "//tools/format/formatters:text",
    ],
)
