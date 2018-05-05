licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "README.md",
])

py_binary(
    name = "photolint",
    srcs = ["photolint.py"],
    default_python_version = "PY3",
    deps = [
        ":lintercache",
        ":linters",
        ":util",
        ":workspace",
    ],
)

py_library(
    name = "workspace",
    srcs = ["workspace.py"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "lintercache",
    srcs = ["lintercache.py"],
    deps = [
        ":linters",
        ":util",
    ],
)

py_library(
    name = "lightroom",
    srcs = ["lightroom.py"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "linters",
    srcs = ["linters.py"],
    deps = [
        ":lightroom",
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_test(
    name = "tests/linters_test",
    size = "small",
    srcs = ["tests/linters_test.py"],
    default_python_version = "PY3",
    deps = [
        ":linters",
    ],
)

py_test(
    name = "tests/workspace_test",
    size = "small",
    srcs = ["tests/workspace_test.py"],
    default_python_version = "PY3",
    deps = [
        ":workspace",
    ],
)
