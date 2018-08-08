# Benchmarks.

py_binary(
    name = "bzip2",
    srcs = ["bzip2.py"],
    data = [
        "@bzip2",
        "@bzip2//:bzip2_srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "bzip2_test",
    srcs = ["bzip2_test.py"],
    deps = [
        ":bzip2",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "llvm_test_suite",
    srcs = ["llvm_test_suite.py"],
    data = [
        "@llvm_test_suite//:queens",
        "@llvm_test_suite//:queens_srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/labm8:bazelutil",
    ],
)
