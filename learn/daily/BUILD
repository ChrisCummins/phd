# Daily programming challenges.

# Targets in this package are prefixed by the letter "d" followed by their date
# in YYMMDD format. The "d" prefix is to ensure that python module names are
# legal.

cc_library(
    name = "d181211_matrix_multiplication",
    srcs = ["d181211_matrix_multiplication.cc"],
    hdrs = ["d181211_matrix_multiplication.h"],
    deps = [
        "//labm8/cpp:logging",
        "@boost//:numeric",
        "@boost//:serialization",
        "@boost//:thread",
    ],
)

cc_test(
    name = "d181211_matrix_multiplication_test",
    srcs = ["d181211_matrix_multiplication_test.cc"],
    deps = [
        ":d181211_matrix_multiplication",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "d181213_kruskal_spanning_tree",
    srcs = ["d181213_kruskal_spanning_tree.cc"],
    hdrs = ["d181213_kruskal_spanning_tree.h"],
    deps = [
        "//labm8/cpp:logging",
        "@boost//:graph",
        "@boost//:numeric",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "d181213_kruskal_spanning_tree_test",
    srcs = ["d181213_kruskal_spanning_tree_test.cc"],
    deps = [
        ":d181213_kruskal_spanning_tree",
        "//labm8/cpp:test",
    ],
)

cc_binary(
    name = "d190815_unique_int",
    srcs = ["d190815_unique_int.cc"],
)
