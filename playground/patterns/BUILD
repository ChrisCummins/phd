cc_test(
    name = "main",
    size = "small",
    srcs = [
        "container.hpp",
        "main.cc",
    ],
    copts = [
        "-Isrc/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = ["//lib/phd:main"],
)
