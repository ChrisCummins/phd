# Datasets for compiler CFG problems.

py_binary(
    name = "import_from_github",
    srcs = ["import_from_github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:reachability_py_pb2",
        "//labm8:lockfile",
        "//labm8:ppar",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "make_opencl_devmap_dataset",
    srcs = ["make_opencl_devmap_dataset.py"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = [
        "//experimental/compilers/reachability:cfg_datasets",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_binary(
    name = "make_vbcd",
    srcs = ["make_vbcd.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":import_from_github",
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//datasets/github/scrape_repos:cloner",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos:importer",
        "//datasets/github/scrape_repos:scraper",
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/proto:scrape_repos_py_pb2",
        "//datasets/linux",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:llvm_util",
        "//experimental/compilers/reachability:reachability_py_pb2",
        "//labm8:decorators",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "make_vbcd_test",
    srcs = ["make_vbcd_test.py"],
    deps = [
        ":make_vbcd",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
