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
    size = 'small',
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
    size = 'small',
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
    size = 'small',
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
    size = 'small',
)

cc_test(
    name =  "011-big-mandelbrot",
    srcs = ["011-big-mandelbrot.cc"],
    copts = ["-Ithird_party/opencl/include"],
    linkopts = select({
        '//:darwin': ['-framework OpenCL'],
        '//conditions:default': ['-lOpenCL'],
    }),
    deps = ["//:opencl"],
    size = 'small',
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
    size = 'small',
)

cc_test(
    name =  "014-palindrome",
    srcs = ["014-palindrome.c"],
    args = ['abcdefedcba'],  # palindrome
    size = 'small',
)

cc_test(
    name =  "015-longest-path-matrix",
    srcs = ["015-longest-path-matrix.cc"],
    size = 'small',
)

sh_test(
    name = "016-min-hops",
    srcs = ["016-min-hops.py"],
    size = 'small',
)

sh_test(
    name = "017-longest-substrings",
    srcs = ["017-longest-substrings.py"],
    size = 'small',
)

sh_test(
    name = "018-list-binary-tree",
    srcs = ["018-list-binary-tree.py"],
    size = 'small',
)

sh_test(
    name = "019-validate-bst",
    srcs = ["019-validate-bst.py"],
    size = 'small',
)

sh_test(
    name = "020-highest-product",
    srcs = ["020-highest-product.py"],
    size = 'small',
)

cc_test(
    name = "021-string-reverse-cpp",
    srcs = ["021-string-reverse.cc"],
    size = 'small',
)

sh_test(
    name = "021-string-reverse-py",
    srcs = ["021-string-reverse.py"],
    size = 'small',
)

sh_binary(
    name = "022-multiplication-table",
    srcs = ["022-multiplication-table.py"],
)

sh_binary(
    name = "023-all-subsets",
    srcs = ["023-all-subsets.py"],
)

cc_test(
    name = "024-linked-list-cycle",
    srcs = ["024-linked-list-cycle.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest//:main",
    ],
    size = 'small',
)

sh_binary(
    name = "025-first-nonrepeated-char",
    srcs = ["025-first-nonrepeated-char.py"],
)

cc_binary(
    name = "026-remove-dupes",
    srcs = ["026-remove-dupes.cc"],
)

sh_test(
    name = "027-missing-number",
    srcs = ["027-missing-number.py"],
    size = 'small',
)
