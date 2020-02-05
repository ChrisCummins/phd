# This package defines a Python library for accessing the MNIST dataset.

load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

pybind_extension(
    "mnist_data_loader",
    srcs = ["mnist_data_loader.cc"],
    deps = [
        "//datasets/mnist/cpp:mnist_data_loader",
    ],
)

py_test(
    name = "mnist_data_loader_test",
    srcs = ["mnist_data_loader_test.py"],
    data = [":mnist_data_loader.so"],
    deps = [
        "//labm8/py:test",
    ],
)
