cc_library(
    name = "shared",
    srcs = glob(["src/*.c"]),
    hdrs = glob([
        "include/*.h",
        "include/*.hpp",
    ]),
    copts = [
        "-Ilearn/hoocl/include",
    ],
    deps = ["//third_party/opencl"],
)

cc_test(
    name = "001-dev-info",
    srcs = ["001-dev-info.cc"],
    copts = [
        "-Ilearn/hoocl/include",
    ],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = [":shared"],
)

cc_test(
    name = "002-vadd",
    srcs = ["002-vadd.c"],
    copts = [
        "-Ilearn/hoocl/include",
    ],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = [":shared"],
)

cc_test(
    name = "003-vadd_cpp",
    srcs = ["003-vadd_cpp.cc"],
    copts = [
        "-Ilearn/hoocl/include",
    ],
    data = [":vadd_kernel"],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = [":shared"],
)

filegroup(
    name = "vadd_kernel",
    srcs = ["vadd.cl"],
)
