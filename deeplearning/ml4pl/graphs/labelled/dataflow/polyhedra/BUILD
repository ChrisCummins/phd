# Annotating program graphs with polyhedral SCoPs

py_library(
    name = "polyhedra",
    srcs = ["polyhedra.py"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:graph_builder",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/cfg:llvm_util",
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
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:graph_builder",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
