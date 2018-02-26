cc_binary(
    name = "binary_search_tree",
    srcs = ["binary_search_tree.cc"],
)

cc_test(
    name = "breadth_first_search",
    size = "small",
    srcs = [
        "breadth_first_search.cc",
        "graph.h",
    ],
)

sh_test(
    name = "bts_rotate",
    size = "small",
    srcs = ["bts_rotate.py"],
)

cc_test(
    name = "depth_first_search",
    size = "small",
    srcs = [
        "depth_first_search.cc",
        "graph.h",
    ],
)

sh_test(
    name = "graph",
    size = "small",
    srcs = ["graph.py"],
)

sh_test(
    name = "search",
    size = "small",
    srcs = ["search.py"],
)

cc_test(
    name = "priority_queues",
    size = "small",
    srcs = ["priority_queues.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest//:main",
    ],
)
