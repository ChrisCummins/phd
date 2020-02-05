cc_library(
    name = "backprop",
    hdrs = ["backprop.h"],
    deps = [
        ":elementwise_subtraction",
    ],
)

py_test(
    name = "batch_linear_regression_test",
    size = "small",
    srcs = ["batch_linear_regression_test.py"],
    deps = [
        "//labm8/py:test",
    ],
)

cc_library(
    name = "elementwise_addition",
    hdrs = ["elementwise_addition.h"],
)

cc_library(
    name = "elementwise_subtraction",
    hdrs = ["elementwise_subtraction.h"],
)

cc_library(
    name = "feed_forward",
    hdrs = ["feed_forward.h"],
    deps = [
        ":elementwise_addition",
        ":matrix_vector_multiply",
        ":sigmoid",
        ":softmax",
    ],
)

cc_library(
    name = "matrix_vector_multiply",
    hdrs = ["matrix_vector_multiply.h"],
)

cc_binary(
    name = "mnist_neural_network",
    srcs = ["mnist_neural_network.cc"],
    deps = [
        ":elementwise_addition",
        ":feed_forward",
        ":matrix_vector_multiply",
        ":randomness",
        ":softmax",
        "//datasets/mnist/cpp:mnist_data_loader",
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
    ],
)

cc_library(
    name = "randomness",
    hdrs = ["randomness.h"],
)

cc_library(
    name = "sigmoid",
    hdrs = ["sigmoid.h"],
)

cc_library(
    name = "softmax",
    hdrs = ["softmax.h"],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    deps = [
        ":softmax",
        "//labm8/cpp:test",
    ],
)
