cc_library(
    name = "main",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/rt/*.h"]),
    copts = ["-Iexperimental/graphics/rt/include"] + select({
        "//:darwin": ["-Iexternal/tbb_mac/include"],
        "//conditions:default": ["-Iexternal/tbb_lin/include"],
    }),
    visibility = ["//experimental/graphics/rt:__subpackages__"],
    deps = select({
        "//:darwin": ["@tbb_mac//:main"],
        "//conditions:default": ["@tbb_lin//:main"],
    }),
)
