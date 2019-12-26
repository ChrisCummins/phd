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
        "//deeplearning/ml4pl/graphs:graph_viz",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:graph_builder",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/cfg:control_flow_graph",
        "//deeplearning/ml4pl/models:log_analysis",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/ggnn",
        "//deeplearning/ncc/inst2vec:api",
        "//research/grewe_2013_cgo:feature_extractor",
    ],
)
