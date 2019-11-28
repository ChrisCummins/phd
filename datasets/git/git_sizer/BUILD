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
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:system",
    ],
)
