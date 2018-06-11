# Experiments in machine learning for compiler test case analysis.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "export_clang_opencl_dataset",
    srcs = ["export_clang_opencl_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
        requirement("mysqlclient"),
        requirement("humanize"),
        requirement("absl-py"),
    ],
)

py_binary(
    name = "prepare_discriminator_dataset",
    srcs = ["prepare_discriminator_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
        requirement("humanize"),
        requirement("absl-py"),
    ],
)

py_binary(
    name = "train_discriminator",
    srcs = ["train_discriminator.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        requirement("absl-py"),
        requirement("humanize"),
        requirement("keras"),
        requirement("numpy"),
    ],
)
