# This package creates labelled graphs for the POJ-104 aglorithm classification
# task.

py_binary(
    name = "make_classifyapp_dataset",
    srcs = ["make_classifyapp_dataset.py"],
    data = ["//deeplearning/ncc/published_results:dictionary"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/bytecode:splitters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:database_exporters",
        "//deeplearning/ncc/inst2vec:api",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "make_classifyapp_dataset_test",
    srcs = ["make_classifyapp_dataset_test.py"],
    data = [
        "//deeplearning/ml4pl/graphs/labelled/classifyapp/test_data:bytecode",
        "//deeplearning/ncc/published_results:dictionary",
    ],
    deps = [
        ":make_classifyapp_dataset",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:bazelutil",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)
