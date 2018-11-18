# Tools for managing machines.

py_binary(
    name = "data_tiers",
    srcs = ["data_tiers.py"],
    deps = [
        "//lib/labm8:pbutil",
        "//system/machines/proto:data_tiers_pb2_py",
    ],
)

py_library(
    name = "mirrored_directory",
    srcs = ["mirrored_directory.py"],
    deps = [
        "//system/machines/proto:machine_spec_pb2_py",
        "//third_party/py/absl",
        "//lib/labm8:labtypes",
    ],
    visibility=["//system/machines:__subpackages__"],
)

py_test(
    name = "mirrored_directory_test",
    srcs = ["mirrored_directory_test.py"],
    deps = [
        ":mirrored_directory",
        "//lib/labm8:fs",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
