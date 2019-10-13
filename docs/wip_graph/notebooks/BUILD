# Jupyter notebook with project dependencies.

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/adversary:opencl_deadcode_inserter",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:control_and_data_flow_graph",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:reachability_pb_py",
        "//research/grewe_2013_cgo:feature_extractor",
    ],
)
