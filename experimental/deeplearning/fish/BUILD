# Experiments in machine learning for compiler test case analysis.

py_binary(
    name = "export_clang_opencl_dataset",
    srcs = ["export_clang_opencl_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//lib/labm8:fs",
        "//lib/labm8:pbutil",
    ],
)

py_binary(
    name = "prepare_discriminator_dataset",
    srcs = ["prepare_discriminator_dataset.py"],
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
