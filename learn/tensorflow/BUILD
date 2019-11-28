# Learning to use TensorFlow

py_binary(
    name = "mnist_regression",
    srcs = ["mnist_regression.py"],
    deps = [
        "//labm8/py:app",
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
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_test(
    name = "conditionals_and_loops_test",
    srcs = ["conditionals_and_loops_test.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/tensorflow",
    ],
)
