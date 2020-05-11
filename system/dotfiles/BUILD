# My dotfiles and configuration.

load("@cec_exports_repo//tools/bzl:exports_repo.bzl", "exports_repo")

exports_repo(
    name = "export",
    path_rename = {
        "system/dotfiles/README.md": "README.md",
    },
    remote = "git@github.com:ChrisCummins/dotfiles.git",
    tags = ["manual"],
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
