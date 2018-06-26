# A package for listing available OpenCL devices.

cc_binary(
    name = "clinfo",
    srcs = ["clinfo.cc"],
    deps = [":libclinfo"],
)

cc_library(
    name = "libclinfo",
    srcs = [
        "libclinfo.cc",
        "libclinfo.h",
    ],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//third_party/opencl",
    ],
)
