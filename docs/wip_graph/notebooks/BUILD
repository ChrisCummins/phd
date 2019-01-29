# Jupyter notebook with project dependencies.

py_binary(
    name = "notebooks",
    srcs = ["//notebooks"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/adversary:opencl_deadcode_inserter",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//experimental/compilers/reachability:cfg_datasets",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//gpu/portable_mapping_of_data_parallel_programs_to_opencl:feature_extractor",
    ],
)
