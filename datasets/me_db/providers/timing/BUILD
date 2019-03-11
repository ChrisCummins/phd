# Package to handle data from Timing.app.

py_binary(
    name = "timing",
    srcs = ["timing.py"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:humanize",
        "//labm8:pbutil",
    ],
)
