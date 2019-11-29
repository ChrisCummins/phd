# This package constructs program graphs from LLVM intermediate representation.

py_library(
    name = "llvm_statements",
    srcs = ["llvm_statements.py"],
    deps = [
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs:graph_iterators",
        "//deeplearning/ml4pl/graphs:graph_query",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//deeplearning/ml4pl/graphs/unlabelled/cg:call_graph",
        "//deeplearning/ncc:rgx_utils",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "llvm2graph",
    srcs = ["llvm2graph.py"],
    deps = [
        ":graph_builder",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
    ],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
    ],
)
