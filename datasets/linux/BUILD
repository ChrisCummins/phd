# A dataset of the linux sources.

py_library(
    name = "linux",
    srcs = ["linux.py"],
    data = [
        "//datasets/linux/generated_headers",
        "@linux_srcs//:hdrs",
        "@linux_srcs//:includes",
        "@linux_srcs//:srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:decorators",
        "//labm8:fs",
    ],
)

py_test(
    name = "linux_test",
    srcs = ["linux_test.py"],
    deps = [
        ":linux",
        "//labm8:app",
        "//labm8:test",
    ],
)
