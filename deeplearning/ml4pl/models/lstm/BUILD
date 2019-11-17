py_library(
    name = "bytecode2seq",
    srcs = ["bytecode2seq.py"],
    data = [
        ":llvm_vocab",
        "//deeplearning/clgen/corpuses/lexer:lexer_worker",
    ],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "bytecode2seq_test",
    srcs = ["bytecode2seq_test.py"],
    deps = [
        ":bytecode2seq",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
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

py_binary(
    name = "encoded_bytecode_database",
    srcs = ["encoded_bytecode_database.py"],
    deps = [
        "//labm8:app",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
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
        ":bytecode2seq",
        ":graph2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
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
    visibility = ["//visibility:public"],
    deps = [
        ":graph2seq",
        ":lstm_utils",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:base_utils",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "lstm_node_classifier",
    srcs = ["lstm_node_classifier.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph2seq",
        ":lstm_utils",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:base_utils",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "lstm_graph_classifier_smoke_test",
    size = "large",
    srcs = ["lstm_graph_classifier_smoke_test.py"],
    deps = [
        ":lstm_graph_classifier",
        "//deeplearning/ml4pl/models:classifier_smoke_tester",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "lstm_utils",
    srcs = ["lstm_utils.py"],
    deps = [
        "//deeplearning/ml4pl/models:classifier_base",
        "//labm8:app",
        "//third_party/py/keras",
    ],
)

py_binary(
    name = "populate_encoded_bytecode_database",
    srcs = ["populate_encoded_bytecode_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":encoded_bytecode_database",
        ":graph2seq",
        ":lstm_utils",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple:graph_batcher",
        "//deeplearning/ml4pl/models:base_utils",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
