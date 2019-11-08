# Annotating graphs with alias sets.

py_library(
    name = "polyhedra",
    srcs = ["polyhedra.py"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8:app",
        "//labm8:decorators",
    ],
)

py_test(
    name = "polyhedra_test",
    srcs = ["polyhedra_test.py"],
    deps = [
        ":polyhedra",
        "//compilers/llvm:clang",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
