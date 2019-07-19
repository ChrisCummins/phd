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
        ":conftest",
        "//third_party/py/git",
    ]
)
py_binary(
    name = "export_source_tree",
    srcs = ["export_source_tree.py"],
    data = ["//tools/source_tree/data:bazel_wrapper"],
    visibility = ["//visibility:public"],
    deps = [
        "//:getconfig",
        "//datasets/github:api",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//third_party/py/git",
        "//third_party/py/github",
    ],
)

sh_test(
    name = "export_source_tree_smoke_test",
    timeout = "long",
    srcs = ["export_source_tree_smoke_test.sh"],
    data = [":export_source_tree"],
)
