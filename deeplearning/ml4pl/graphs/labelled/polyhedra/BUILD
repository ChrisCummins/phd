# Annotating program graphs with polyhedral SCoPs

py_library(
    name = "polyhedra",
    srcs = ["polyhedra.py"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8/py:app",
        "//labm8/py:decorators",
    ],
)

py_test(
    name = "polyhedra_test",
    srcs = ["polyhedra_test.py"],
    deps = [
        ":polyhedra",
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
