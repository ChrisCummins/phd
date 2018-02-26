cc_library(
    name = "main",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/phd/*"]),
    copts = [
        "-Isrc/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@benchmark//:main",
        "@gtest//:main",
    ],
)
