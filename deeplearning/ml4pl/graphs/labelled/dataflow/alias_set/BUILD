# Annotating graphs with alias sets.

py_library(
    name = "alias_set",
    srcs = ["alias_set.py"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        "//compilers/llvm:opt_util",
        "//labm8/py:app",
        "//labm8/py:decorators",
    ],
)

py_test(
    name = "alias_set_test",
    srcs = ["alias_set_test.py"],
    deps = [
        ":alias_set",
        "//compilers/llvm:clang",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)
