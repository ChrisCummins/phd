cc_test(
    name = "main",
    size = "small",
    srcs = [
        "lm.cc",
        "meta-math.cc",
        "tests.hpp",
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
