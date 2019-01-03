# A dataset of OpenCL Heterogeneous Device Mappings.

licenses(["restricted"])

filegroup(
    name = "csv_files",
    srcs = [
        "amd.csv",
        "nvidia.csv",
    ],
)

py_library(
    name = "opencl_device_mapping_dataset",
    srcs = ["opencl_device_mapping_dataset.py"],
    data = [":csv_files"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deeptune/opencl/adversary:opencl_deadcode_inserter",
        "//labm8:bazelutil",
        "//labm8:decorators",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "opencl_device_mapping_dataset_test",
    srcs = ["opencl_device_mapping_dataset_test.py"],
    deps = [
        ":opencl_device_mapping_dataset",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)
