cc_library(
    name = "analysis",
    srcs = ["analysis.cc"],
    hdrs = ["analysis.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":reachability",
        "//labm8/cpp:status",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_library(
    name = "data_flow_pass",
    srcs = ["data_flow_pass.cc"],
    hdrs = ["data_flow_pass.h"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_library(
    name = "domtree",
    srcs = ["domtree.cc"],
    hdrs = ["domtree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_library(
    name = "reachability",
    srcs = ["reachability.cc"],
    hdrs = ["reachability.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_test(
    name = "reachability_test",
    srcs = ["reachability_test.cc"],
    deps = [
        ":reachability",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
    ],
)
