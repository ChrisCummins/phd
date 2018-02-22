py_binary(
    name = "photolint",
    srcs = ["photolint.py"],
    default_python_version = "PY3",
    deps = [
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
    name = "lightroom",
    srcs = ["lightroom.py"],
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
    name = "test_workspace",
    srcs = ["test_workspace.py"],
    default_python_version = "PY3",
    deps = [
        ":workspace",
    ],
)
