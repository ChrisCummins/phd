# A package for listing available OpenCL devices.

cc_binary(
    name = "clinfo",
    srcs = ["clinfo.cc", "clinfo.h"],
    deps = ["//third_party/opencl", "//gpu/clinfo/proto:clinfo_cc_pb2"],
    linkopts = ["-framework OpenCL"],
)
