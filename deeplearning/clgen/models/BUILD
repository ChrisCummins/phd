# This package implements the machine learning models for CLgen.

py_library(
    name = "backends",
    srcs = ["backends.py"],
    deps = [
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//labm8:cache",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "builders",
    srcs = ["builders.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "builders_test",
    srcs = ["builders_test.py"],
    deps = [
        ":builders",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "data_generators",
    srcs = ["data_generators.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_generators_test",
    srcs = ["data_generators_test.py"],
    deps = [
        ":data_generators",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "keras_backend",
    srcs = ["keras_backend.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backends",
        ":builders",
        ":data_generators",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:crypto",
        "//labm8:labdate",
        "//labm8:logutil",
        "//labm8:pbutil",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "keras_backend_test",
    size = "enormous",
    srcs = ["keras_backend_test.py"],
    deps = [
        ":keras_backend",
        ":models",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:crypto",
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        ":keras_backend",
        ":tensorflow_backend",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:crypto",
        "//labm8:lockfile",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pretrained",
    srcs = ["pretrained.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        ":keras_backend",
        ":tensorflow_backend",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:cache",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:crypto",
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "tensorflow_backend",
    srcs = ["tensorflow_backend.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backends",
        ":builders",
        ":data_generators",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:crypto",
        "//labm8:labdate",
        "//labm8:logutil",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "tensorflow_backend_test",
    size = "enormous",
    srcs = ["tensorflow_backend_test.py"],
    deps = [
        ":models",
        ":tensorflow_backend",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:crypto",
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
