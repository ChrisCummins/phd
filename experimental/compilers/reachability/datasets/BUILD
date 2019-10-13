# Datasets for compiler CFG problems.

py_binary(
    name = "import_from_github",
    srcs = ["import_from_github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:random_opt_pb_py",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:lockfile",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "import_from_poj104",
    srcs = ["import_from_poj104.py"],
    deps = [
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:reachability_pb_py",
        "//labm8:app",
        "//labm8:fs",
    ],
)

py_library(
    name = "linux",
    srcs = ["linux.py"],
    deps = [
        ":opencl",
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//datasets/linux",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:llvm_util",
        "//experimental/compilers/reachability:random_opt_pb_py",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/pyparsing",
    ],
)

py_test(
    name = "linux_test",
    srcs = ["linux_test.py"],
    deps = [
        ":linux",
        "//datasets/linux",
        "//labm8:app",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "make_opencl_devmap_dataset",
    srcs = ["make_opencl_devmap_dataset.py"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = [
        ":linux",
        ":opencl",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//labm8:app",
        "//labm8:prof",
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
        ":linux",
        ":opencl",
        "//compilers/llvm:opt",
        "//datasets/github:api",
        "//datasets/github/scrape_repos:cloner",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos:importer",
        "//datasets/github/scrape_repos:scraper",
        "//datasets/github/scrape_repos/preprocessors",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:llvm_util",
        "//experimental/compilers/reachability:random_opt_pb_py",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:llvm_util",
        "//experimental/compilers/reachability:random_opt_pb_py",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "opencl_test",
    srcs = ["opencl_test.py"],
    deps = [
        ":opencl",
        "//labm8:app",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)
