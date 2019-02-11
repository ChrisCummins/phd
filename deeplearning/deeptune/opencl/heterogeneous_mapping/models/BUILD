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

py_test(
    name = "deeptune_test",
    srcs = ["deeptune_test.py"],
    deps = [
        ":deeptune",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8:test",
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

py_test(
    name = "grewe_test",
    srcs = ["grewe_test.py"],
    deps = [
        ":grewe",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8:test",
    ],
)

py_library(
    name = "lda",
    srcs = ["lda.py"],
    deps = [
        ":base",
        ":ncc",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:utils",
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
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//experimental/compilers/reachability:llvm_util",
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
        "//labm8:test",
    ],
)

py_library(
    name = "ncc",
    srcs = ["ncc.py"],
    data = [
        "//deeplearning/ncc/published_results:embeddings",
        "//deeplearning/ncc/published_results:task_devmap_kernels",
        "//deeplearning/ncc/published_results:vocabulary",
    ],
    visibility = ["//visibility:public"],
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

py_test(
    name = "ncc_test",
    srcs = ["ncc_test.py"],
    deps = [
        ":ncc",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8:test",
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

py_test(
    name = "static_mapping_test",
    srcs = ["static_mapping_test.py"],
    deps = [
        ":static_mapping",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8:test",
    ],
)

py_library(
    name = "testlib",
    testonly = 1,
    srcs = ["testlib.py"],
    deps = [
        "//third_party/py/absl",
    ],
)
