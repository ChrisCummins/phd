# A non-bazel python 2/3 compatabile module for invoking linters on project
# sources.

py_library(
    name = "linters_lib",
    srcs = ["linters_lib.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "linters_lib_test",
    srcs = ["linters_lib_test.py"],
    deps = [
        ":linters_lib",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
