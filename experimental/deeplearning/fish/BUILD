# Experiments in machine learning for compiler test case analysis.

py_binary(
    name = "export_clang_opencl_dataset",
    srcs = ["export_clang_opencl_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/mysql",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        ":train_discriminator",
        "//deeplearning/clgen",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:viz",
    ],
)

py_binary(
    name = "prepare_discriminator_dataset",
    srcs = ["prepare_discriminator_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:labtypes",
        "//labm8:pbutil",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "train_discriminator",
    srcs = ["train_discriminator.py"],
    deps = [
        "//deeplearning/clgen",
        "//experimental/deeplearning/fish/proto:fish_py_pb2",
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)
