# lmk - An alarm clock for the command line.

licenses(["notice"])  # MIT

exports_files([
    "demo.png",
    "LICENSE",
    "README.md",
])

py_binary(
    name = "lmk",
    srcs = ["lmk.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/humanize",
    ],
)
