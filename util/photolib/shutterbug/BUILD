# Shutterbug is a utility for creating DVD archive backups of photo libraries.

py_library(
    name = "shutterbug",
    srcs = ["shutterbug.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "shutterbug_test",
    srcs = ["shutterbug_test.py"],
    deps = [
        ":shutterbug",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "pack",
    srcs = ["pack.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":shutterbug",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "unpack",
    srcs = ["unpack.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":shutterbug",
        "//third_party/py/absl",
    ],
)
