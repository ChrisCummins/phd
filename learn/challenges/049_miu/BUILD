cc_binary(
    name = "049_miu",
    srcs = ["049_miu.cc"],
    deps = [
        ":miu",
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
    ],
)

cc_library(
    name = "miu",
    srcs = ["miu.cc"],
    hdrs = ["miu.h"],
    deps = [
        "//labm8/cpp:string",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "miu_test",
    srcs = ["miu_test.cc"],
    deps = [
        ":miu",
        "//labm8/cpp:string",
        "//labm8/cpp:test",
    ],
)
