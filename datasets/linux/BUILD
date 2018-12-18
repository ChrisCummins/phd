# A dataset of the linux sources.

py_library(
    name = "linux",
    srcs = ["linux.py"],
    data = [
        "@linux_srcs//:includes",
        "@linux_srcs//:kernel_c_hdrs",
        "@linux_srcs//:kernel_c_srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:bazelutil",
        "//labm8:decorators",
        "//labm8:fs",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "linux_test",
    srcs = ["linux_test.py"],
    deps = [
        ":linux",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
