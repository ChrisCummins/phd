# My dotfiles and configuration.

filegroup(
    name = "dotfiles",
    srcs = glob(
        [
            "**/*",
        ],
        exclude = [
            "EXPORT.py",
        ],
    ),
)

py_binary(
    name = "EXPORT",
    srcs = ["EXPORT.py"],
    deps = ["//tools/source_tree:export_source_tree"],
)
