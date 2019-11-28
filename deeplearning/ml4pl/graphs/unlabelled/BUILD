# Unlabelled program graphs.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_binary(
    name = "make_unlabelled_graph_dataset",
    srcs = ["make_unlabelled_graph_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:database_exporters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8/py:app",
        "//labm8/py:prof",
    ],
)

proto_library(
    name = "programl_pb",
    srcs = ["programl.proto"],
)

python_proto_library(
    name = "programl_pb_py",
    visibility = ["//visibility:public"],
    deps = [":programl_pb"],
)
