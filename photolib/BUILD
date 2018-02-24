py_binary(
    name = "photolint",
    srcs = ["photolint.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":util",
    ],
)

py_library(
    name = "lintercache",
    srcs = ["lintercache.py"],
    srcs_version = "PY3",
    deps = [
        ":linters",
        ":util",
    ],
)

py_library(
    name = "lightroom",
    srcs = ["lightroom.py"],
    srcs_version = "PY3",
    deps = [
        ":util",
    ],
)

py_library(
    name = "linters",
    srcs = ["linters.py"],
    srcs_version = "PY3",
    deps = [
        ":lightroom",
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tests/linters_test",
    size = "small",
    srcs = ["tests/linters_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linters",
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
    ],
)
