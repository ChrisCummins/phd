cc_test(
    name = "main",
    size = "small",
    srcs = [
        "benchmarks.cc",
        "benchmarks.hpp",
    ],
    copts = [
        "-Ilab/lm/include",
        "-Isrc/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "//lab/lm:main",
        "//src/phd:main",
    ],
)
