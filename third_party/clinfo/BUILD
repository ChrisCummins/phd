# Query OpenCL platform and device parameters.
#
# Author: Simon Leblanc <https://github.com/simleb>
# From: https://github.com/simleb/clinfo

licenses(["restricted"])  # GPL v3

filegroup(
    name = "clinfo_c_src",
    srcs = ["clinfo.c"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "clinfo",
    srcs = [":clinfo_c_src"],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
)
