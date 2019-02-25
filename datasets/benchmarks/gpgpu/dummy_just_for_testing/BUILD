# A dummy benchmark suite for testing.

filegroup(
    name = "hello_sources",
    srcs = ["hello.cc"],
    visibility = ["//datasets/benchmarks/gpgpu:__subpackages__"],
)

cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
    copts = [
        "-Ilearn/hoocl/include",
    ],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
)
