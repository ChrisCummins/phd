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
    linkopts = ["-ldl"],
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "@libopencl//:libOpenCL",
        "//third_party/opencl",
    ],
)
