# Experiments in machine learning for compiler test case analysis.

py_binary(
    name = "export_clang_opencl_dataset",
    srcs = ["export_clang_opencl_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_pb_py",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//third_party/py/mysql",
    ],
)

py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    deps = [
        ":train_discriminator",
        "//deeplearning/clgen",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:viz",
    ],
)

py_binary(
    name = "prepare_discriminator_dataset",
    srcs = ["prepare_discriminator_dataset.py"],
    deps = [
        "//experimental/deeplearning/fish/proto:fish_pb_py",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "train_discriminator",
    srcs = ["train_discriminator.py"],
    deps = [
        "//deeplearning/clgen",
        "//experimental/deeplearning/fish/proto:fish_pb_py",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)
