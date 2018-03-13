licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

cc_library(
    name = "libcecl",
    srcs = ["libcecl.c"],
    hdrs = ["libcecl.h"],
    copts = ["-Ithird_party/opencl/include"],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "mkcecl",
    srcs = ["mkcecl.sh"],
)

sh_binary(
    name = "runcecl",
    srcs = ["runcecl.sh"],
)
