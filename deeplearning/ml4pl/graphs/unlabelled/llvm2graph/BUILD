# This package constructs program graphs from LLVM intermediate representation.

py_library(
    name = "call_graph",
    srcs = ["call_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:labtypes",
        "//third_party/py/networkx",
        "//third_party/py/pydot",
        "//third_party/py/pyparsing",
    ],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":call_graph",
        ":node_encoder",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs:nx_utils",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/cfg:llvm_util",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_builder_test",
    srcs = ["graph_builder_test.py"],
    deps = [
        ":graph_builder",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_builder_regression_test",
    size = "enormous",
    srcs = ["graph_builder_regression_test.py"],
    data = ["//deeplearning/ml4pl/testing/data/bytecode_regression_tests:bytecodes"],
    deps = [
        ":graph_builder",
        "//labm8/py:fs",
        "//labm8/py:test",
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
    name = "llvm_statements",
    srcs = ["llvm_statements.py"],
    deps = [
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/cfg:llvm_util",
        "//deeplearning/ncc:rgx_utils",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "node_encoder",
    srcs = ["node_encoder.py"],
    data = [
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/node_embeddings:dictionary",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph/node_embeddings:embeddings",
    ],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//third_party/py/numpy",
    ],
)
