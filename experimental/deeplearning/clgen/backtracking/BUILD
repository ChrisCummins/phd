# A CLgen model which uses backtracking during sampling.

py_binary(
    name = "backtracking",
    srcs = ["backtracking.py"],
    deps = [
        ":backtracking_model",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/preprocessors",
        "//research/cummins_2017_cgo:generative_model",
        "//third_party/py/absl",
    ],
)

sh_test(
    name = "backtracking_smoke_test",
    srcs = ["backtracking_smoke_test.sh"],
    data = [
        ":backtracking",
    ],
)

py_library(
    name = "backtracking_db",
    srcs = ["backtracking_db.py"],
    deps = [
        "//labm8:sqlutil",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "backtracking_model",
    srcs = ["backtracking_model.py"],
    deps = [
        ":logger",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/models:tensorflow_backend",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:labdate",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "backtracking_model_test",
    srcs = ["backtracking_model_test.py"],
    deps = [
        ":backtracking_model",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    deps = [
        ":backtracking_db",
        "//third_party/py/absl",
    ],
)
