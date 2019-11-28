# Producing a labelled dataset for heterogeneous device mapping from PACT'17
# data.

py_binary(
    name = "make_devmap_dataset",
    srcs = ["make_devmap_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ncc/inst2vec:api",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)
