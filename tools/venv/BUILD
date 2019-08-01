# TODO(github.com/ChrisCummins/phd/issues/49): Work in progress.

load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

constraint_value(
    name = "virtualenv_python2",
    constraint_setting = "@bazel_tools//tools/python:py2_interpreter_path",
)

constraint_value(
    name = "virtualenv_python3",
    constraint_setting = "@bazel_tools//tools/python:py3_interpreter_path",
)

platform(
    name = "virtualenv_platform",
    constraint_values = [
        ":virtualenv_python2",
        ":virtualenv_python3",
    ],
)

py_runtime(
    name = "virtualenv_py2_runtime",
    interpreter_path = "tools/venv/phd/bin/python",
)

py_runtime(
    name = "virtualenv_py3_runtime",
    interpreter_path = "tools/venv/phd/bin/python",
)

py_runtime_pair(
    name = "virtualenv_platform_runtimes",
    py2_runtime = ":virtualenv_py2_runtime",
    py3_runtime = ":virtualenv_py3_runtime",
)

toolchain(
    name = "virtualenv_platform_python_toolchain",
    target_compatible_with = [
        ":virtualenv_python2",
        ":virtualenv_python3",
    ],
    toolchain = ":virtualenv_platform_runtimes",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
