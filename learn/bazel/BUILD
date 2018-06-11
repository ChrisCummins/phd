# Learning to use Bazel.

load("@requirements//:requirements.bzl", "requirement")

py_test(
    name = "python_binary",
    srcs = ["python_binary.py"],
    data = [
        ":data_binary",
        ":data_file",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//lib/labm8:bazelutil",
        requirement("absl-py"),
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
        "//third_party/tensorflow",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
