# Shutterbug is a utility for creating DVD archive backups of photo libraries.

py_library(
    name = "shutterbug",
    srcs = ["shutterbug.py"],
)

py_binary(
    name = "pack",
    srcs = ["pack.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shutterbug"],
)

py_binary(
    name = "unpack",
    srcs = ["unpack.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shutterbug"],
)
