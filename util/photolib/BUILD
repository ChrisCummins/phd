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
        "//labm8/py:app",
        "//labm8/py:decorators",
    ],
)

py_library(
    name = "dashcam",
    srcs = ["dashcam.py"],
    deps = [
        "//labm8/py:app",
    ],
)

par_binary(
    name = "photolib",
    srcs = ["photolib.py"],
    deps = [
        "//labm8/py:app",
    ],
)

par_binary(
    name = "photolib-csv",
    srcs = ["photolib-csv.py"],
    deps = [
        ":contentfiles",
        ":linters",
        ":workspace",
        ":xmp_cache",
        "//labm8/py:app",
    ],
)

par_binary(
    name = "photolib-dashcam-tzoffset",
    srcs = ["photolib-dashcam-tzoffset.py"],
    deps = [
        ":dashcam",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
    ],
)

par_binary(
    name = "photolib-dashcat",
    srcs = ["photolib-dashcat.py"],
    deps = [
        ":dashcam",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
    ],
)

par_binary(
    name = "photolib-lint",
    srcs = ["photolib-lint.py"],
    deps = [
        ":linters",
        ":workspace",
        "//labm8/py:app",
        "//labm8/py:humanize",
    ],
)

py_library(
    name = "workspace",
    srcs = ["workspace.py"],
    deps = [
        ":common",
        "//labm8/py:app",
    ],
)

py_library(
    name = "lintercache",
    srcs = ["lintercache.py"],
    deps = [
        ":common",
        ":workspace",
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:shell",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "xmp_cache",
    srcs = ["xmp_cache.py"],
    deps = [
        ":common",
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
        "//third_party/py/libxmp",
        "//third_party/py/sqlalchemy",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        ":photolib.par",
        ":photolib-csv.par",
        ":photolib-dashcam-tzoffset.par",
        ":photolib-dashcat.par",
        ":photolib-lint.par",
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
        "//labm8/py:app",
        "//labm8/py:shell",
        "//third_party/py/protobuf",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//labm8/py:app",
        "//util/photolib/proto:photolint_pb_py",
    ],
)

py_test(
    name = "tests/linters_test",
    size = "small",
    srcs = ["tests/linters_test.py"],
    deps = [
        ":linters",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_test(
    name = "tests/workspace_test",
    size = "small",
    srcs = ["tests/workspace_test.py"],
    deps = [
        ":workspace",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
