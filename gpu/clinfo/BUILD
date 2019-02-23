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
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//phd:logging",
        "//phd:string",
        "//third_party/opencl",
        "//third_party/opencl:libOpenCL.so",
    ],
)
