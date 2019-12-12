# TODO(github.com/ChrisCummins/ProGraML/issues/28): Fix me.

py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:run",
        "//deeplearning/ml4pl/models/lstm:lstm_utils",
        "//deeplearning/ml4pl/seq:graph2seq",
        "//deeplearning/ml4pl/seq:ir2seq",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:progress",
    ],
)

py_test(
    name = "lstm_test",
    size = "enormous",
    srcs = ["lstm_test.py"],
    deps = [
        ":lstm",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)

py_library(
    name = "lstm_utils",
    srcs = ["lstm_utils.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)
