# Learning to use Bazel.

py_test(
    name = "python_binary",
    srcs = ["python_binary.py"],
    data = [
        ":data_binary",
        ":data_file",
    ],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
    ],
)

filegroup(
    name = "data_file",
    srcs = ["data_file.txt"],
)

cc_binary(
    name = "data_binary",
    srcs = ["data_binary.cc"],
)

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/tensorflow",
    ],
)
