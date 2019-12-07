# This moduel defines sequence representations for programs.

py_library(
    name = "graph2seq",
    srcs = ["graph2seq.py"],
    deps = [
        ":ir2seq",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
        "//third_party/py/lru_dict",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph2seq_test",
    timeout = "long",
    srcs = ["graph2seq_test.py"],
    deps = [
        ":graph2seq",
        ":ir2seq",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "ir2seq",
    srcs = ["ir2seq.py"],
    data = [":llvm_vocab"],
    deps = [
        ":lexers",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ncc:vocabulary",
        "//deeplearning/ncc/inst2vec:api",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "ir2seq_test",
    timeout = "long",
    srcs = ["ir2seq_test.py"],
    deps = [
        ":ir2seq",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "lexers",
    srcs = ["lexers.py"],
    data = [
        "//deeplearning/clgen/corpuses/lexer:lexer_worker",
    ],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lexers_test",
    timeout = "long",
    srcs = ["lexers_test.py"],
    deps = [
        ":lexers",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "llvm_vocab",
    srcs = ["llvm_vocab.json"],
)
