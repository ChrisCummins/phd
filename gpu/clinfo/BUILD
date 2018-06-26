# A package for listing available OpenCL devices.

cc_binary(
    name = "clinfo",
    srcs = ["clinfo.cc"],
    visibility = ["//visibility:public"],
    deps = [":libclinfo"],
)

cc_library(
    name = "libclinfo",
    srcs = [
        "libclinfo.cc",
        "libclinfo.h",
    ],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
