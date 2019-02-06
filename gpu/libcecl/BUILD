licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

filegroup(
    name = "libcecl_header",
    srcs = ["libcecl.h"],
    visibility = ["//visibility:public"],
)

# Two targets for the same code, but for different purposes. :libcecl is a
# library that can be used in the deps attribute of cc_binary targets that need
# to link against libcecl. The :libcecl.so target produces a shared object which
# can be used in the data attribute of other targets.

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

cc_binary(
    name = "libcecl.so",
    srcs = [
        "libcecl.c",
        "libcecl.h",
    ],
    copts = ["-Ithird_party/opencl/include"],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "mkcecl",
    srcs = ["mkcecl.sh"],
)

sh_binary(
    name = "runcecl",
    srcs = ["runcecl.sh"],
    visibility = ["//visibility:public"],
)
