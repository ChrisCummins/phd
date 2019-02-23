licenses(["notice"])  # MIT.

# OpenCL 1.2 headers. Use this instead of system packages.
cc_library(
    name = "opencl",
    hdrs = [
        "include/cl.h",
        "include/cl.hpp",
    ],
    visibility = ["//visibility:public"],
)

# An empty file which can be used when needed to link against OpenCL. Add this
# target to your OpenCL binary's dependencies.
cc_library(
    name = "libOpenCL.so",
    srcs = ["libopencl.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
