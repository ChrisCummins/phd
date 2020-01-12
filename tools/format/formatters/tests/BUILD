# Tests for formatters.

py_test(
    name = "cxx_test",
    srcs = ["cxx_test.py"],
    deps = [
        ":testing",
        "//labm8/py:test",
        "//tools/format/formatters:cxx",
    ],
)

py_test(
    name = "sql_test",
    srcs = ["sql_test.py"],
    deps = [
        ":testing",
        "//labm8/py:test",
        "//tools/format/formatters:sql",
    ],
)

py_library(
    name = "testing",
    testonly = 1,
    srcs = ["testing.py"],
    deps = [
        "//labm8/py:fs",
        "//tools/format:formatter",
    ],
)

py_test(
    name = "text_test",
    srcs = ["text_test.py"],
    deps = [
        ":testing",
        "//labm8/py:test",
        "//tools/format/formatters:text",
    ],
)
