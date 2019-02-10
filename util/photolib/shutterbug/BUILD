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
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "pack",
    srcs = ["pack.py"],
    deps = [
        ":shutterbug",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "unpack",
    srcs = ["unpack.py"],
    deps = [
        ":shutterbug",
        "//third_party/py/absl",
    ],
)
