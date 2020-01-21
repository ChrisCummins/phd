# Tools for working with the source tree.

load("//tools/bzl:exports.bzl", "exports_repo")
load("@subpar//:subpar.bzl", "par_binary")

exports_repo(
    name = "export",
    github_repo = "bazel_subtree_github_export",
    move_file_mapping = {
        "tools/source_tree/README.md": "README.md",
        "tools/source_tree/LICENSE": "LICENSE",
        "tools/source_tree/travis.yml": ".travis.yml",
    },
    targets = ["//tools/source_tree:export_source_tree"],
)

par_binary(
    name = "deploy_pip",
    srcs = ["deploy_pip.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":phd_workspace",
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:fs",
    ],
)

par_binary(
    name = "export_source_tree",
    srcs = ["export_source_tree.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":phd_workspace",
        "//:build_info",
        "//datasets/github:api",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//third_party/py/git",
        "//third_party/py/github",
        "//tools/git:export_subtree",
    ],
)

py_library(
    name = "phd_workspace",
    srcs = ["phd_workspace.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//third_party/py/git",
        "//tools/git:export_subtree",
    ],
)
