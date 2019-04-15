# My solutions to problems from "Cracking the Coding Interview: 150
# Programming Questions and Solutions", Gayle Laakmann McDowell, 2011.

cc_library(
    name = "ctci",
    hdrs = ["ctci.h"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)

cc_test(
    name = "0101-unique-chars-in-string-cpp",
    size = "small",
    srcs = ["0101-unique-chars-in-string.cc"],
    deps = [":ctci"],
)

py_test(
    name = "0101-unique-chars-in-string-py",
    size = "small",
    srcs = ["0101-unique-chars-in-string_test.py"],
    main = "0101-unique-chars-in-string_test.py",
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)

cc_test(
    name = "0102-reverse-string-cpp",
    size = "small",
    srcs = ["0102-reverse-string.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0103-permutations-cpp",
    size = "small",
    srcs = ["0103-permutations.cc"],
    deps = [":ctci"],
)

py_test(
    name = "0103-permutations-py",
    size = "small",
    srcs = ["0103-permutations_test.py"],
    main = "0103-permutations_test.py",
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)

cc_test(
    name = "0104-escape-string-cpp",
    size = "small",
    srcs = ["0104-escape-string.cc"],
    deps = [":ctci"],
)

py_test(
    name = "0104-escape-string-py",
    size = "small",
    srcs = ["0104-escape-string_test.py"],
    main = "0104-escape-string_test.py",
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)

cc_test(
    name = "0105-string-compression-cpp",
    size = "small",
    srcs = ["0105-string-compression.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0107-matrix-zero-cpp",
    size = "small",
    srcs = ["0107-matrix-zero.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0108-string-rotation-cpp",
    size = "small",
    srcs = ["0108-string-rotation.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0201-list-remove-dups-cpp",
    size = "small",
    srcs = ["0201-list-remove-dups.cc"],
    deps = [":ctci"],
)

py_test(
    name = "0201-list-remove-dups-py",
    size = "small",
    srcs = ["0201-list-remove-dups_test.py"],
    main = "0201-list-remove-dups_test.py",
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)

cc_test(
    name = "0202-linked-list-k-last-cpp",
    size = "small",
    srcs = ["0202-linked-list-k-last.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0207-list-palindrone-cpp",
    size = "small",
    srcs = ["0207-list-palindrone.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0302-stack-min-cpp",
    size = "small",
    srcs = ["0302-stack-min.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0303-set-of-stacks-cpp",
    size = "small",
    srcs = ["0303-set-of-stacks.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0305-stack-queue-cpp",
    size = "small",
    srcs = ["0305-stack-queue.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0402-directed-graph-routefinder-cpp",
    size = "small",
    srcs = ["0402-directed-graph-routefinder.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0404-binary-tree-list-cpp",
    size = "small",
    srcs = ["0404-binary-tree-list.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0502-binary-double-cpp",
    size = "small",
    srcs = ["0502-binary-double.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0505-int-bitdiff-cpp",
    size = "small",
    srcs = ["0505-int-bitdiff.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "0508-horiz-line-cpp",
    size = "small",
    srcs = ["0508-horiz-line.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "1101-merge-arrays-cpp",
    size = "small",
    srcs = ["1101-merge-arrays.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "1102-sort-anagrams-cpp",
    size = "small",
    srcs = ["1102-sort-anagrams.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "1301-last-k-lines-cpp",
    size = "small",
    srcs = ["1301-last-k-lines.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "1307-tree-copy-cpp",
    size = "small",
    srcs = ["1307-tree-copy.cc"],
    deps = [":ctci"],
)

# FIXME: this is broken
#
# cc_test(
#     name =  "1310-2d-alloc-cpp",
#     srcs = ["1310-2d-alloc.cc"],
#     deps = [":ctci"],
# )

cc_test(
    name = "1701-num-swap-cpp",
    size = "small",
    srcs = ["1701-num-swap.cc"],
    deps = [":ctci"],
)

cc_test(
    name = "1702-tic-tac-toe-cpp",
    size = "small",
    srcs = ["1702-tic-tac-toe.cc"],
    deps = [":ctci"],
)

py_test(
    name = "1804-number-of-2s-py",
    size = "small",
    srcs = ["1804-number-of-2s_test.py"],
    main = "1804-number-of-2s_test.py",
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)
