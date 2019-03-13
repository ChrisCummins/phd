# Port of "Case Study A" from "//docs/2017_09_pact/code:Case Study A.ipynb".

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    visibility = ["//deeplearning/deeptune/opencl/heterogeneous_mapping:__subpackages__"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "heterogeneous_mapping",
    srcs = ["heterogeneous_mapping.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "heterogeneous_mapping_test",
    size = "enormous",
    srcs = ["heterogeneous_mapping_test.py"],
    deps = [
        ":conftest",
        ":heterogeneous_mapping",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//deeplearning/deeptune/opencl/heterogeneous_mapping:__subpackages__"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:app",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy",
        "//third_party/py/scipy:scikit_learn",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":conftest",
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//labm8:app",
        "//labm8:test",
    ],
)
