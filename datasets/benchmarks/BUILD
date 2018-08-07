# Benchmarks.

py_binary(
    name = "bzip2",
    srcs = ["bzip2.py"],
    data = [
        "@bzip2",
        "@bzip2//:bzip2_srcs",
    ],
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
