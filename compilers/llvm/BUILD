# A python wrapper around LLVM binaries.

py_binary(
    name = "clang",
    srcs = ["clang.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    deps = [
        ":clang",
        ":llvm",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "clang_format",
    srcs = ["clang_format.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang-format",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "clang_format_test",
    srcs = ["clang_format_test.py"],
    deps = [
        ":clang_format",
        ":llvm",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "llvm_as",
    srcs = ["llvm_as.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-as",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-as",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "llvm_link",
    srcs = ["llvm_link.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-link",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-link",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "llvm_link_test",
    srcs = ["llvm_link_test.py"],
    deps = [
        ":llvm",
        ":llvm_link",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "llvm",
    srcs = ["llvm.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "opt",
    srcs = ["opt.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux//:opt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "opt_test",
    srcs = ["opt_test.py"],
    deps = [
        ":llvm",
        ":opt",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
