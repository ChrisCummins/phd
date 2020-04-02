load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

py_library(
    name = "analysis",
    srcs = ["analysis.py"],
    data = [":analysis_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
    ],
)

pybind_extension(
    name = "analysis_pybind",
    srcs = ["analysis_pybind.cc"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/graph:program_graph_builder",
        "//programl/graph/analysis",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)
