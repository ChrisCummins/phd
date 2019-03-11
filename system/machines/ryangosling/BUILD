# Scripts and data for machine ryangosling.

filegroup(
    name = "ryangosling_proto",
    srcs = ["ryangosling.pbtxt"],
)

filegroup(
    name = "ryangosling_zsh",
    srcs = ["ryangosling.zsh"],
)

py_test(
    name = "ryangosling_test",
    srcs = ["ryangosling_test.py"],
    data = [":ryangosling_proto"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
        "//system/machines:machine",
        "//system/machines:mirrored_directory",
    ],
)
