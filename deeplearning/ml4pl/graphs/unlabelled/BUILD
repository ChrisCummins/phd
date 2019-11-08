# Unlabelled program graphs.

py_binary(
    name = "make_unlabelled_graph_dataset",
    srcs = ["make_unlabelled_graph_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:database_exporters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8:app",
        "//labm8:prof",
    ],
)
