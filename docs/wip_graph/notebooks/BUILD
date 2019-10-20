# Jupyter notebook with project dependencies.

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ncc/inst2vec:api",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/labelled/graph_dict",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/reachability:make_reachability_dataset",
        "//research/grewe_2013_cgo:feature_extractor",
    ],
)
