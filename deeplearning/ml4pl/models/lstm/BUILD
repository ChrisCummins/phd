py_library(
    name = "bytecode2seq",
    srcs = ["bytecode2seq.py"],
    data = ["//deeplearning/clgen/corpuses/lexer:lexer_worker"],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "derive_vocabulary",
    srcs = ["derive_vocabulary.py"],
    deps = [
        ":bytecode2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:jsonutil",
        "//labm8:prof",
        "//labm8:sqlutil",
    ],
)

py_library(
    name = "graph2seq",
    srcs = ["graph2seq.py"],
    data = [":llvm_vocab"],
    deps = [
        ":bytecode2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:jsonutil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph2seq_test",
    srcs = ["graph2seq_test.py"],
    deps = [
        ":graph2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

filegroup(
    name = "llvm_vocab",
    srcs = ["llvm_vocab.json"],
)

py_binary(
    name = "lstm_graph_classifier",
    srcs = ["lstm_graph_classifier.py"],
    deps = [
        ":bytecode2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
