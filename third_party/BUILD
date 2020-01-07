# Third party code.

sh_binary(
    name = "oclgrind",
    srcs = ["oclgrind.sh"],
    data = select({
        "//:darwin": ["@oclgrind_mac//:oclgrind"],
        "//conditions:default": ["@oclgrind_linux//:oclgrind"],
    }),
    visibility = ["//visibility:public"],
)
