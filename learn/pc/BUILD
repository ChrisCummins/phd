cc_test(
    name = "001-fib",
    size = "small",
    srcs = ["001-fib.cc"],
    copts = [
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest//:main",
    ],
)

cc_test(
    name = "002-linked-lists",
    size = "small",
    srcs = ["002-linked-lists.cc"],
    copts = [
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest//:main",
    ],
)

cc_test(
    name = "003-string-inc",
    size = "small",
    srcs = ["003-string-inc.cc"],
    copts = [
        "-Ilib/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = ["//lib/phd:main"],
)

cc_test(
    name = "004-fractal",
    size = "small",
    srcs = ["004-fractal.cc"],
    copts = [
        "-Ilib/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = ["//lib/phd:main"],
)

cc_binary(
    name = "humble-001",
    srcs = ["humble-001.cc"],
    args = [
        "-f",
        "1",
        "-g",
        "2",
    ],
)

cc_binary(
    name = "humble-003-vector",
    srcs = ["humble-003-vector.cc"],
)
