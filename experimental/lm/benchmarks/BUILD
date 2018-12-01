cc_test(
    name = "main",
    size = "small",
    srcs = [
        "benchmarks.cc",
        "benchmarks.hpp",
    ],
    copts = [
        "-Iexperimental/lm/include",
        "-Ilib/phd/include",
    ],
    deps = [
        "//experimental/lm:main",
        "//lib/phd:main",
    ],
)
