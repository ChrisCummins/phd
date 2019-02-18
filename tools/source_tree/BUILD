# Tools for working with the source tree.

py_binary(
    name = "export_source_tree",
    srcs = ["export_source_tree.py"],
    data = ["//tools/source_tree/data:bazel_wrapper"],
    deps = [
        "//config:getconfig",
        "//datasets/github:api",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//third_party/py/absl",
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
