# This package implements the machine learning models for CLgen.

py_library(
    name = "builders",
    srcs = ["builders.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "builders_test",
    srcs = ["builders_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builders",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
    ],
)

py_library(
    name = "data_generators",
    srcs = ["data_generators.py"],
    deps = [
        "//deeplearning/clgen/corpuses",
    ],
)

py_test(
    name = "data_generators_test",
    srcs = ["data_generators_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_generators",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
    ],
)

py_library(
    name = "keras_backend",
    srcs = ["keras_backend.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        ":data_generators",
        ":models",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:labdate",
        "//lib/labm8:lockfile",
        "//lib/labm8:logutil",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "keras_backend_test",
    size = "enormous",
    srcs = ["keras_backend_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_backend",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//lib/labm8:crypto",
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
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
    ],
)
