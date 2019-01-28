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

py_binary(
    name = "04_train_model",
    srcs = ["04_train_model.py"],
    deps = [
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//experimental/compilers/reachability:graph_model",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/graph_nets",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)
