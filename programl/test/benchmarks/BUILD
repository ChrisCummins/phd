cc_test(
    name = "01_llvm2graph",
    srcs = ["01_llvm2graph.cc"],
    deps = [
        "//labm8/cpp:bazelutil",
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//programl/ir/llvm",
        "//programl/proto:program_graph_cc",
        "//third_party/llvm",
        "@boost//:filesystem",
    ],
)

py_test(
    name = "pipeline",
    srcs = ["pipeline.py"],
    data = [
        "//programl/test/data:llvm_ir",
    ],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//programl/graph/analysis/py:analysis",
        "//programl/graph/py:graph_tuple_builder",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/ir/llvm/py:llvm",
    ],
)

py_test(
    name = "legacy_pipeline",
    srcs = ["legacy_pipeline.py"],
    data = [
        "//programl/test/data:llvm_ir",
    ],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/reachability",
        "//deeplearning/ml4pl/graphs/llvm2graph:llvm2graph_py",
        "//deeplearning/ml4pl/graphs/llvm2graph:node_encoder",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:prof",
    ],
)

py_binary(
    name = "read_graph_tuples",
    srcs = ["read_graph_tuples.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:prof",
        "//programl/graph/analysis/py:analysis",
        "//programl/graph/py:graph_tuple_builder",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/ir/llvm/py:llvm",
    ],
)
