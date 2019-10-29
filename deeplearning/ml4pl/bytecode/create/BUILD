# Datasets for compiler CFG problems.

py_binary(
    name = "import_from_github",
    srcs = ["import_from_github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:lockfile",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "import_from_linux",
    srcs = ["import_from_linux.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//datasets/linux",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/pyparsing",
    ],
)

py_test(
    name = "import_from_linux_test",
    srcs = ["import_from_linux_test.py"],
    deps = [
        ":import_from_linux",
        "//datasets/linux",
        "//labm8:app",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "import_from_npb",
    srcs = ["import_from_npb.py"],
    deps = [
        "//compilers/llvm:llvm_dis",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:sqlutil",
    ],
)

py_library(
    name = "import_from_pact17_devmap",
    srcs = ["import_from_pact17_devmap.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:opt_util",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
        "//labm8:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "import_from_pact17_devmap_test",
    srcs = ["import_from_pact17_devmap_test.py"],
    deps = [
        ":import_from_pact17_devmap",
        "//labm8:app",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "import_from_poj104",
    srcs = ["import_from_poj104.py"],
    deps = [
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8:app",
        "//labm8:fs",
    ],
)

py_binary(
    name = "populate_cfg_table",
    srcs = ["populate_cfg_table.py"],
    deps = [
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
    ],
)
