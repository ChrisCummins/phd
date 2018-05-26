# This package implements the machine learning models for CLgen.

py_library(
    name = "builders",
    srcs = ["builders.py"],
    deps = [
        "//deeplearning/clgen/proto:model_py_pb2",
    ],
)


py_test(
    name = "builders_test",
    srcs = ["builders_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builders",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:corpuses",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:languages",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:cache",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
        "//lib/labm8:labdate",
        "//lib/labm8:labtypes",
        "//lib/labm8:lockfile",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
    ],
)
