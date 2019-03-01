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
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//third_party/opencl",
        "//phd:string",
        "//phd:logging",
        "//phd:statusor",
        "@com_google_absl//absl/strings",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_test(
    name = "libclinfo_test",
    srcs = ["libclinfo_test.cc"],
    deps = [
        ":libclinfo",
        "//phd:test",
        "//third_party/opencl",
    ],
)
