py_library(
    name = "dpack",
    srcs = ["dpack.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//lib/dpack/proto:dpack_pb2_py",
    ],
)

py_test(
    name = "dpack_test",
    size = "small",
    srcs = ["dpack_test.py"],
    deps = [
        ":dpack",
    ],
)
