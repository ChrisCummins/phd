load("@requirements//:requirements.bzl", "requirement")

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "README.md",
])

py_binary(
    name = "photolint",
    srcs = ["photolint.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":common",
        ":lintercache",
        ":linters",
        ":workspace",
    ],
)

py_library(
    name = "workspace",
    srcs = ["workspace.py"],
    deps = [
        ":common",
    ],
)

py_library(
    name = "lintercache",
    srcs = ["lintercache.py"],
    deps = [
        ":common",
        ":linters",
        requirement("absl-py"),
        requirement("sqlalchemy"),
    ],
)

py_library(
    name = "lightroom",
    srcs = ["lightroom.py"],
    deps = [
        ":common",
        requirement("absl-py"),
        requirement("python-xmp-toolkit"),
        requirement("sqlalchemy"),
        requirement("pytz"),
    ],
)

py_library(
    name = "linters",
    srcs = ["linters.py"],
    deps = [
        ":common",
        ":lightroom",
        requirement("absl-py"),
        requirement("protobuf"),
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//util/photolib/proto:photolint_py_pb2",
        requirement("absl-py"),
    ],
)

py_test(
    name = "tests/linters_test",
    size = "small",
    srcs = ["tests/linters_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linters",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_test(
    name = "tests/workspace_test",
    size = "small",
    srcs = ["tests/workspace_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":workspace",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
