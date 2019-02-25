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

cc_binary(
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

cc_binary(
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

cc_binary(
    name = "003-vadd_cpp",
    srcs = ["003-vadd_cpp.cc"],
    copts = [
        "-Ilearn/hoocl/include",
    ],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = [":shared"],
)
