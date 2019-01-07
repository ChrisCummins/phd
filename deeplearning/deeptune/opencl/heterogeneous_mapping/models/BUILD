# Models for OpenCL hetereogeneous device mapping.

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "deeptune",
    srcs = ["deeptune.py"],
    deps = [
        ":base",
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/absl",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "grewe",
    srcs = ["grewe.py"],
    deps = [
        ":base",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/scipy:scikit_learn",
    ],
)

py_library(
    name = "lda",
    srcs = ["lda.py"],
    deps = [
        ":base",
        ":ncc",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/ncc:task_utils",
        "//deeplearning/ncc:vocabulary",
        "//experimental/compilers/reachability:llvm_util",
        "//third_party/py/absl",
        "//third_party/py/graph_nets",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "lda_test",
    srcs = ["lda_test.py"],
    deps = [
        ":models",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:utils",
        "//labm8:test",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    data = [
        "//deeplearning/ncc/published_results:embeddings",
        "//deeplearning/ncc/published_results:task_devmap_kernels",
        "//deeplearning/ncc/published_results:vocabulary",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":deeptune",
        ":grewe",
        ":lda",
        ":ncc",
        ":static_mapping",
        "//labm8:labtypes",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "models_test",
    timeout = "long",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:utils",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "ncc",
    srcs = ["ncc.py"],
    deps = [
        ":base",
        ":deeptune",
        "//compilers/llvm:clang",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ncc:task_utils",
        "//deeplearning/ncc:vocabulary",
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "static_mapping",
    srcs = ["static_mapping.py"],
    deps = [
        ":base",
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/absl",
        "//third_party/py/pandas",
    ],
)
