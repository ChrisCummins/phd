# Datasets for compiler CFG problems.

py_binary(
    name = "import_from_directory",
    srcs = ["import_from_directory.py"],
    deps = [
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
    ],
)

py_binary(
    name = "import_from_github",
    srcs = ["import_from_github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:lockfile",
        "//labm8/py:sqlutil",
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
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/pyparsing",
    ],
)

# TODO(github.com/ChrisCummins/ProGraML/issues/7): Linux build is broken.
#py_test(
#    name = "import_from_linux_test",
#    srcs = ["import_from_linux_test.py"],
#    deps = [
#        ":import_from_linux",
#        "//datasets/linux",
#        "//labm8/py:app",
#        "//labm8/py:system",
#        "//labm8/py:test",
#        "//third_party/py/pytest",
#    ],
#)

py_binary(
    name = "import_from_npb",
    srcs = ["import_from_npb.py"],
    deps = [
        "//compilers/llvm:llvm_dis",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
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
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "import_from_pact17_devmap_test",
    srcs = ["import_from_pact17_devmap_test.py"],
    deps = [
        ":import_from_pact17_devmap",
        "//labm8/py:app",
        "//labm8/py:system",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "import_from_poj104",
    srcs = ["import_from_poj104.py"],
    deps = [
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
    ],
)
