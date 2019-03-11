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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    deps = [
        ":clang",
        ":llvm",
        "//labm8:app",
        "//labm8:test",
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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "clang_format_test",
    srcs = ["clang_format_test.py"],
    deps = [
        ":clang_format",
        ":llvm",
        "//labm8:app",
        "//labm8:test",
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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "llvm_link_test",
    srcs = ["llvm_link_test.py"],
    deps = [
        ":llvm",
        ":llvm_link",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "llvm",
    srcs = ["llvm.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "opt_test",
    srcs = ["opt_test.py"],
    deps = [
        ":llvm",
        ":opt",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        ":llvm_as",
        ":opt",
        "//labm8:app",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":llvm",
        ":util",
        "//labm8:app",
        "//labm8:test",
    ],
)
