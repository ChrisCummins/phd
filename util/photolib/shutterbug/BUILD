# Shutterbug is a utility for creating DVD archive backups of photo libraries.

py_library(
    name = "shutterbug",
    srcs = ["shutterbug.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_test(
    name = "shutterbug_test",
    srcs = ["shutterbug_test.py"],
    deps = [
        ":shutterbug",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "pack",
    srcs = ["pack.py"],
    deps = [
        ":shutterbug",
        "//labm8/py:app",
    ],
)

py_binary(
    name = "unpack",
    srcs = ["unpack.py"],
    deps = [
        ":shutterbug",
        "//labm8/py:app",
    ],
)
