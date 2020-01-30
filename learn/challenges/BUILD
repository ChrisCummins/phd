cc_test(
    name = "001-int-average",
    size = "small",
    srcs = ["001-int-average.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)

cc_binary(
    name = "006-gray-code",
    srcs = ["006-gray-code.cc"],
)

cc_test(
    name = "008-linked-list",
    size = "small",
    srcs = ["008-linked-list.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest",
    ],
)

cc_test(
    name = "009-longest-substr",
    size = "small",
    srcs = ["009-longest-substr.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)

cc_test(
    name = "010-inc-int-arr",
    size = "small",
    srcs = ["010-inc-int-arr.cc"],
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "011-big-mandelbrot",
    size = "small",
    srcs = ["011-big-mandelbrot.cc"],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    deps = ["//third_party/opencl"],
)

cc_test(
    name = "012-simple-cyphers",
    size = "small",
    srcs = ["012-simple-cyphers.cc"],
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "014-palindrome",
    size = "small",
    srcs = ["014-palindrome.c"],
    args = ["abcdefedcba"],  # palindrome
)

cc_test(
    name = "015-longest-path-matrix",
    size = "small",
    srcs = ["015-longest-path-matrix.cc"],
)

py_test(
    name = "016-min-hops",
    size = "small",
    srcs = ["016-min-hops.py"],
)

py_test(
    name = "017-longest-substrings",
    size = "small",
    srcs = ["017-longest-substrings.py"],
)

py_test(
    name = "018-list-binary-tree",
    size = "small",
    srcs = ["018-list-binary-tree.py"],
)

py_test(
    name = "019-validate-bst",
    size = "small",
    srcs = ["019-validate-bst.py"],
)

py_test(
    name = "020-highest-product",
    size = "small",
    srcs = ["020-highest-product.py"],
)

cc_test(
    name = "021-string-reverse-cpp",
    size = "small",
    srcs = ["021-string-reverse.cc"],
)

py_test(
    name = "021-string-reverse-py",
    size = "small",
    srcs = ["021-string-reverse.py"],
    main = "021-string-reverse.py",
)

py_test(
    name = "022-multiplication-table",
    srcs = ["022-multiplication-table.py"],
)

py_test(
    name = "023-all-subsets",
    srcs = ["023-all-subsets.py"],
)

cc_test(
    name = "024-linked-list-cycle",
    size = "small",
    srcs = ["024-linked-list-cycle.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest",
    ],
)

py_test(
    name = "025-first-nonrepeated-char",
    srcs = ["025-first-nonrepeated-char.py"],
)

cc_binary(
    name = "026-remove-dupes",
    srcs = ["026-remove-dupes.cc"],
)

py_test(
    name = "027-missing-number",
    size = "small",
    srcs = ["027-missing-number.py"],
)

cc_test(
    name = "029-unique-substrings",
    size = "small",
    srcs = ["029-unique-substrings.cc"],
    deps = ["@gtest"],
)

cc_binary(
    name = "030_reverse_words_in_sentence",
    srcs = ["030_reverse_words_in_sentence.cc"],
)

cc_test(
    name = "031_matrix_diagonal",
    srcs = ["031_matrix_diagonal.cc"],
    copts = ["-fopenmp"],
    # macOS: Requires libomp Homebrew package.
    linkopts = [
        "-fopenmp",
    ] + select({
        "//:darwin": [
            "-L/usr/local/Cellar/libomp/10.0.0/lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "032_offset_date",
    srcs = ["032_offset_date.cc"],
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "033_tree_sibling",
    srcs = ["033_tree_sibling.cc"],
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "034_shared_ptr",
    srcs = ["034_shared_ptr.cc"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "035_estimate_derivative",
    srcs = ["035_estimate_derivative.cc"],
    deps = [
        "//labm8/cpp:test",
    ],
)

cc_test(
    name = "036_hamming_distance",
    srcs = ["036_hamming_distance.cc"],
    deps = [
        "//labm8/cpp:logging",
    ],
)

cc_test(
    name = "037_gradient_descent",
    srcs = ["037_gradient_descent.cc"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:test",
    ],
)
