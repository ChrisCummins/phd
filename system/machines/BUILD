# Tools for managing machines.

load("@subpar//:subpar.bzl", "par_binary")

py_binary(
    name = "data_tiers",
    srcs = ["data_tiers.py"],
    deps = [
        "//labm8:pbutil",
        "//system/machines/proto:data_tiers_pb_py",
    ],
)

par_binary(
    name = "machine",
    srcs = ["machine.py"],
    visibility = ["//system/machines:__subpackages__"],
    deps = [
        ":mirrored_directory",
        "//labm8:app",
        "//labm8:pbutil",
        "//system/machines/proto:machine_spec_pb_py",
    ],
)

py_test(
    name = "machine_test",
    srcs = ["machine_test.py"],
    deps = [
        ":machine",
        "//labm8:app",
        "//labm8:test",
        "//system/machines/proto:machine_spec_pb_py",
    ],
)

py_library(
    name = "mirrored_directory",
    srcs = ["mirrored_directory.py"],
    visibility = ["//system/machines:__subpackages__"],
    deps = [
        "//labm8:app",
        "//labm8:fs",
        "//labm8:labtypes",
        "//system/machines/proto:machine_spec_pb_py",
    ],
)

py_test(
    name = "mirrored_directory_test",
    srcs = ["mirrored_directory_test.py"],
    deps = [
        ":mirrored_directory",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:test",
    ],
)
