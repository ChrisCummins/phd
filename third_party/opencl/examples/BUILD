filegroup(
    name = "hello_sources",
    srcs = ["hello.cc"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "hello",
    srcs = [":hello_sources"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
