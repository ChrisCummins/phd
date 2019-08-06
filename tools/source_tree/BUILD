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

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    deps = [
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/git",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "conftest_test",
    srcs = ["conftest_test.py"],
    deps = [
        ":conftest",
        "//labm8:test",
        "//third_party/py/git",
    ],
)

par_binary(
    name = "deploy_pip",
    srcs = ["deploy_pip.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":phd_workspace",
        "//:getconfig",
        "//labm8:app",
        "//labm8:fs",
    ],
)

par_binary(
    name = "export_source_tree",
    srcs = ["export_source_tree.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":phd_workspace",
        ":source_tree",
        "//:getconfig",
        "//datasets/github:api",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:humanize",
        "//third_party/py/git",
        "//third_party/py/github",
    ],
)

py_library(
    name = "phd_workspace",
    srcs = ["phd_workspace.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":source_tree",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:humanize",
        "//third_party/py/git",
    ],
)

py_library(
    name = "source_tree",
    srcs = ["source_tree.py"],
    deps = [
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/git",
    ],
)

py_test(
    name = "source_tree_test",
    srcs = ["source_tree_test.py"],
    deps = [
        ":conftest",
        ":source_tree",
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/git",
    ],
)
