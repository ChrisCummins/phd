# A set of tools for managing my library of photographs.

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "contentfiles",
    srcs = ["contentfiles.py"],
    deps = [
        ":xmp_cache",
        "//labm8:app",
        "//labm8:decorators",
    ],
)

py_binary(
    name = "photolint",
    srcs = ["photolint.py"],
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
