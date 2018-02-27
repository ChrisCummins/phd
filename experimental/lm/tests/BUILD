cc_test(
    name = "main",
    size = "small",
    srcs = [
        "lm.cc",
        "meta-math.cc",
        "tests.hpp",
    ],
    copts = [
        "-Iexperimental/lm/include",
        "-Ilib/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "//experimental/lm:main",
        "//lib/phd:main",
    ],
)
