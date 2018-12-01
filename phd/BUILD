# Core C++ libraries for the phd repo.

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//phd/private:macros_impl",
    ],
)

cc_test(
    name = "macros_test",
    srcs = ["macros_test.cc"],
    deps = [
        ":macros",
        ":test",
    ],
)

cc_library(
    name = "test",
    hdrs = ["test.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)
