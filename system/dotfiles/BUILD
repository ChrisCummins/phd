# My dotfiles and configuration.

load("//tools/bzl:exports.bzl", "exports_repo")

exports_repo(
    name = "export",
    github_repo = "dotfiles",
    move_file_mapping = {
        "system/dotfiles/README.md": "README.md",
    },
    targets = [
        "//system/dotfiles",
    ],
)

filegroup(
    name = "dotfiles",
    srcs = glob([
        "**/*",
    ]),
)
