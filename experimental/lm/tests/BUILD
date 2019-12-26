cc_test(
    name = "lm",
    size = "small",
    srcs = [
        "lm.cc",
    ],
    copts = [
        "-Iexperimental/lm/include",
    ],
    deps = [
        ":tests",
        "//experimental/lm:main",
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "meta_math",
    size = "small",
    srcs = [
        "meta-math.cc",
    ],
    copts = [
        "-Iexperimental/lm/include",
    ],
    deps = [
        ":tests",
        "//experimental/lm:main",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "tests",
    hdrs = ["tests.hpp"],
)
