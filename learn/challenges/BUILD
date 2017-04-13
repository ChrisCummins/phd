cc_test(
    name =  "001-int-average",
    srcs = ["001-int-average.cc"],
    copts = [
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "@gtest//:main",
        "@benchmark//:main",
    ],
)

cc_binary(
    name =  "006-gray-code",
    srcs = ["006-gray-code.cc"],
)

cc_test(
    name =  "008-linked-list",
    srcs = ["008-linked-list.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest//:main",
    ],
)

cc_test(
    name =  "009-longest-substr",
    srcs = ["009-longest-substr.cc"],
    copts = [
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "@gtest//:main",
        "@benchmark//:main",
    ],
)

cc_test(
    name =  "010-inc-int-arr",
    srcs = ["010-inc-int-arr.cc"],
    copts = [
        "-Isrc/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "//src/phd:main",
    ],
)

cc_test(
    name =  "011-big-mandelbrot",
    srcs = ["011-big-mandelbrot.cc"],
    copts = ["-Ithird_party/opencl/include"],
    linkopts = ['$(opencl_ld)'],
    deps = ["//:opencl"],
)

cc_test(
    name =  "012-simple-cyphers",
    srcs = ["012-simple-cyphers.cc"],
    copts = [
        "-Isrc/phd/include",
        "-Iexternal/gtest/include",
        "-Iexternal/benchmark/include",
    ],
    deps = [
        "//src/phd:main",
    ],
)

cc_test(
    name =  "014-palindrome",
    srcs = ["014-palindrome.c"],
    args = ['abcdefedcba'],  # palindrome
)
