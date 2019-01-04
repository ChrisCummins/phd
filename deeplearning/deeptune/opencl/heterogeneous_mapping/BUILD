# Port of "Case Study A" from "//docs/2017_09_pact/code:Case Study A.ipynb".

py_binary(
    name = "heterogeneous_mapping",
    srcs = ["heterogeneous_mapping.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":models",
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "heterogeneous_mapping_test",
    size = "enormous",
    srcs = ["heterogeneous_mapping_test.py"],
    deps = [
        ":heterogeneous_mapping",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:labtypes",
        "//third_party/py/absl",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/scipy:scikit_learn",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//labm8:test",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/scipy:scikit_learn",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
