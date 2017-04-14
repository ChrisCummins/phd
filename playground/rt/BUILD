cc_library(
    name = "main",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/rt/*.h"]),
    copts = ["-Iplayground/rt/include"] + select({
        "//:darwin": ["-Iexternal/tbb_mac/include"],
        "//conditions:default": ["-Iexternal/tbb_lin/include"]
    }),
    deps = select({
        "//:darwin": ["@tbb_mac//:main"],
        "//conditions:default": ["@tbb_lin//:main"],
    }),
    visibility = ["//playground/rt:__subpackages__"],
)
