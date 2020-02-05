# This package defines a C++ library for accessing the MNIST dataset.

cc_library(
    name = "mnist_data_loader",
    srcs = ["mnist_data_loader.cc"],
    hdrs = ["mnist_data_loader.h"],
    data = [
        "//datasets/mnist:mnist_test_images",
        "//datasets/mnist:mnist_test_labels",
        "//datasets/mnist:mnist_train_images",
        "//datasets/mnist:mnist_train_labels",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:bazelutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
    ],
)

cc_test(
    name = "mnist_data_loader_test",
    srcs = ["mnist_data_loader_test.cc"],
    deps = [
        ":mnist_data_loader",
        "//labm8/cpp:test",
    ],
)
