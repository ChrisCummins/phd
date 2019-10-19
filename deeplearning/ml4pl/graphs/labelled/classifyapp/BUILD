# This package creates labelled graphs for the POJ-104 aglorithm classification
# task.

py_binary(
    name = "make_classifyapp_dataset",
    srcs = ["make_classifyapp_dataset.py"],
    deps = [
        "//third_party/py/networkx",
        "//deeplearning/ncc/inst2vec:api",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/bytecode:splitters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:database_exporters",
        "//labm8:app",
        "//labm8:fs",
    ]
)

py_test(
    name = "make_classifyapp_dataset_test",
    srcs = ["make_classifyapp_dataset_test.py"],
    deps = [
        ":make_classifyapp_dataset",
        "//third_party/py/networkx",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:test",
    ]
)
