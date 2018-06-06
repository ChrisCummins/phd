# Experiments in machine learning for compiler test case analysis.

py_binary(
    name = "export_discriminator_data",
    srcs = ["export_discriminator_data.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "train_discriminator",
    srcs = ["train_discriminator.py"],
    deps = ["//experimental/deeplearning/fish/proto:fish_py_pb2"],
)
