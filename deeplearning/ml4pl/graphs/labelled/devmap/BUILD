# Producing a labelled dataset for heterogeneous device mapping from PACT'17
# data.

py_binary(
    name = "make_devmap_dataset",
    srcs = ["make_devmap_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ncc/inst2vec:api",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "make_devmap_dataset_test",
    srcs = ["make_devmap_dataset_test.py"],
    deps = [
        ":make_devmap_dataset",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/testing:random_log_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)
