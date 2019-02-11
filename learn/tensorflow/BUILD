# Learning to use TensorFlow

py_binary(
    name = "mnist_regression",
    srcs = ["mnist_regression.py"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "mnist_regression_test",
    srcs = ["mnist_regression_test.py"],
    deps = [
        ":mnist_regression",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "conditionals_and_loops_test",
    srcs = ["conditionals_and_loops_test.py"],
    deps = [
        "//third_party/py/absl",
        "//labm8:test",
        "//third_party/py/tensorflow",
    ],
)
