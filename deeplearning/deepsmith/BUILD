licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "client",
    srcs = ["client.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db"],
)

py_library(
    name = "datastore",
    srcs = ["datastore.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":client",
        ":db",
        ":generator",
        ":harness",
        ":language",
        ":profiling_event",
        ":result",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/protos:deepsmith",
        "@labm8//:crypto",
    ],
)

py_library(
    name = "db",
    srcs = ["db.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
)

py_test(
    name = "db_test",
    size = "small",
    srcs = ["db_test.py"],
    deps = [
        ":db",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db"],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db", ":testcase"],
)

py_library(
    name = "language",
    srcs = ["language.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [":db"],
)

py_library(
    name = "profiling_event",
    srcs = ["profiling_event.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":client",
        ":db",
        ":result",
        ":testcase",
    ],
)

py_library(
    name = "result",
    srcs = ["result.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":testbed",
        ":testcase",
    ],
)

py_library(
    name = "testbed",
    srcs = ["testbed.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":language",
    ],
)

py_library(
    name = "testcase",
    srcs = ["testcase.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":generator",
        ":harness",
        ":language",
    ],
)
