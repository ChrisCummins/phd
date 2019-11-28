# Hemetic libraries for formatting code.

py_library(
    name = "sql_formatter",
    srcs = ["sql_formatter.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/sqlparse",
    ],
)

py_test(
    name = "sql_formatter_test",
    srcs = ["sql_formatter_test.py"],
    deps = [
        ":sql_formatter",
        "//labm8/py:test",
    ],
)
