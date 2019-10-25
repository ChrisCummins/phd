# Jupyter notebook with project dependencies.

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs:graph_viz",
        "//deeplearning/ml4pl/graphs/labelled/graph_dict",
        "//deeplearning/ml4pl/graphs/labelled/reachability:make_reachability_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/ggnn:ggnn_base",
        "//deeplearning/ml4pl/models/ggnn:ggnn_node_classifier",
        "//deeplearning/ml4pl/models/ggnn:ggnn_utils",
        "//deeplearning/ml4pl/models/ggnn:graph_batcher",
        "//deeplearning/ncc/inst2vec:api",
        "//research/grewe_2013_cgo:feature_extractor",
    ],
)
