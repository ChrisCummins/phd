# Tools for working with the source tree.

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

py_binary(
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

sh_test(
    name = "export_source_tree_self_hosting_smoke_test",
    timeout = "long",
    srcs = ["export_source_tree_self_hosting_smoke_test.sh"],
    data = [":export_source_tree"],
)

py_library(
    name = "phd_workspace",
    srcs = ["phd_workspace.py"],
    deps = [
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

py_binary(
    name = "EXPORT",
    srcs = ["EXPORT.py"],
    deps = [":export_source_tree"],
)
