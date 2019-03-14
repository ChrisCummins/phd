# Oclgrind test data.

cc_binary(
    name = "oclgrind_working",
    testonly = 1,
    srcs = ["oclgrind_working.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    visibility = ["//gpu/oclgrind:__subpackages__"],
    deps = [
        "//phd:logging",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
