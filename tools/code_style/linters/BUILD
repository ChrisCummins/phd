# A non-bazel python 2/3 compatabile module for invoking linters on project
# sources.

py_library(
    name = "linters_lib",
    srcs = ["linters_lib.py"],
)

py_test(
    name = "linters_lib_test",
    srcs = ["linters_lib_test.py"],
    deps = [
        ":linters_lib",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
