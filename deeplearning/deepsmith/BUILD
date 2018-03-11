licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    testonly = 1,
    deps = [
        ":datastore",
        ":db",
        "//deeplearning/deepsmith/proto:datastore",
    ],
)

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
        ":profiling_event",
        ":result",
        ":testbed",
        ":testcase",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith",
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
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_test(
    name = "generator_test",
    size = "small",
    srcs = ["generator_test.py"],
    deps = [
        ":datastore",
        ":db",
        ":generator",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_test(
    name = "harness_test",
    size = "small",
    srcs = ["harness_test.py"],
    deps = [
        ":datastore",
        ":db",
        ":harness",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_library(
    name = "toolchain",
    srcs = ["toolchain.py"],
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
    ],
)

py_library(
    name = "result",
    srcs = ["result.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":profiling_event",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_library(
    name = "result_test",
    srcs = ["result_test.py"],
    deps = [
        ":client",
        ":datastore",
        ":db",
        ":generator",
        ":harness",
        ":profiling_event",
        ":result",
        ":testbed",
        ":testcase",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_library(
    name = "testbed",
    srcs = ["testbed.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        ":db",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith",
    ],
)

py_test(
    name = "testbed_test",
    size = "small",
    srcs = ["testbed_test.py"],
    deps = [
        ":datastore",
        ":db",
        ":testbed",
        ":toolchain",
        "//deeplearning/deepsmith/proto:deepsmith",
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
        ":profiling_event",
        ":toolchain",
    ],
)
