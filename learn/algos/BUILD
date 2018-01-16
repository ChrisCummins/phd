cc_binary(
    name = "binary_search_tree",
    srcs = ["binary_search_tree.cc"],
)

cc_test(
    name =  "breadth_first_search",
    srcs = [
        "breadth_first_search.cc",
        "graph.h",
    ],
    size = 'small',
)

sh_test(
    name = "bts_rotate",
    srcs = ["bts_rotate.py"],
    size = 'small',
)

cc_test(
    name =  "depth_first_search",
    srcs = [
        "depth_first_search.cc",
        "graph.h",
    ],
    size = 'small',
)

sh_test(
    name = "graph",
    srcs = ["graph.py"],
    size = 'small',
)

sh_test(
    name = "search",
    srcs = ["search.py"],
    size = 'small',
)

cc_test(
    name = "priority_queues",
    srcs = ["priority_queues.cc"],
    copts = [
        "-Iexternal/gtest/include",
    ],
    deps = [
        "@gtest//:main",
    ],
    size = 'small',
)
