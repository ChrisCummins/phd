# A tool for creating structured packages of data files.

py_binary(
    name = "dpack",
    srcs = ["dpack.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//labm8:crypto",
        "//labm8:fs",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//system/dpack/proto:dpack_pb2_py",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "dpack_test",
    size = "small",
    srcs = ["dpack_test.py"],
    deps = [
        ":dpack",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
