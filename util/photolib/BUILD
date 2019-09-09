# A set of tools for managing my library of photographs.

licenses(["notice"])  # Apache 2.0

load("@subpar//:subpar.bzl", "par_binary")
load("//tools/bzl:exports.bzl", "exports_repo")

exports_files([
    "LICENSE",
    "README.md",
])

exports_repo(
    name = "export",
    github_repo = "photolib",
    move_file_mapping = {
        "util/photolib/README.md": "README.md",
        "util/photolib/LICENSE": "LICENSE",
        "util/photolib/travis.yml": ".travis.yml",
    },
    targets = [
        "//util/photolib/...",
    ],
)

py_library(
    name = "contentfiles",
    srcs = ["contentfiles.py"],
    deps = [
        ":xmp_cache",
        "//labm8:app",
        "//labm8:decorators",
    ],
)

py_library(
    name = "dashcam",
    srcs = ["dashcam.py"],
    deps = [
        "//labm8:app",
    ]
)

par_binary(
    name = "photolib",
    srcs = ["photolib.py"],
    deps = [
        "//labm8:app",
    ],
)

par_binary(
    name = "photolib-dashcat",
    srcs = ["photolib-dashcat.py"],
    deps = [
        ":dashcam",
        "//labm8:app",
        "//labm8:labtypes",
        "//labm8:humanize",
    ],
)

par_binary(
    name = "photolib-lint",
    srcs = ["photolib-lint.py"],
    deps = [
        ":linters",
        ":workspace",
        "//labm8:app",
        "//labm8:humanize",
    ],
)

py_library(
    name = "workspace",
    srcs = ["workspace.py"],
    deps = [
        ":common",
        "//labm8:app",
    ],
)

py_library(
    name = "lintercache",
    srcs = ["lintercache.py"],
    deps = [
        ":common",
        ":workspace",
        "//:build_info",
        "//labm8:app",
        "//labm8:shell",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "xmp_cache",
    srcs = ["xmp_cache.py"],
    deps = [
        ":common",
        "//:build_info",
        "//labm8:app",
        "//labm8:sqlutil",
        "//third_party/py/libxmp",
        "//third_party/py/sqlalchemy",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        ":photolib-lint.par",
        ":photolib-dashcat.par",
        ":photolib.par",
    ],
)

py_library(
    name = "linters",
    srcs = ["linters.py"],
    deps = [
        ":common",
        ":contentfiles",
        ":lintercache",
        ":xmp_cache",
        "//labm8:app",
        "//labm8:shell",
        "//third_party/py/protobuf",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//labm8:app",
        "//util/photolib/proto:photolint_pb_py",
    ],
)

py_test(
    name = "tests/linters_test",
    size = "small",
    srcs = ["tests/linters_test.py"],
    deps = [
        ":linters",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_test(
    name = "tests/workspace_test",
    size = "small",
    srcs = ["tests/workspace_test.py"],
    deps = [
        ":workspace",
        "//labm8:app",
        "//labm8:test",
    ],
)
