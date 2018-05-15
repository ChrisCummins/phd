# Shutterbug is a utility for creating DVD archive backups of photo libraries.

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "shutterbug",
    srcs = ["shutterbug.py"],
)

py_binary(
    name = "pack",
    srcs = ["pack.py"],
    deps = [":shutterbug"],
)

py_binary(
    name = "unpack",
    srcs = ["pack.py"],
    deps = [":shutterbug"],
)
