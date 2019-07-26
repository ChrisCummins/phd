py_binary(
    name = "git_sizer",
    srcs = ["git_sizer.py"],
    data = select({
        "//:darwin": [
            "@git_sizer_mac//:git-sizer",
        ],
        "//conditions:default": [
            "@git_sizer_linux//:git-sizer",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:system",
    ],
)
