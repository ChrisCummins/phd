# A CLgen model which uses backtracking during sampling.

py_binary(
    name = "backtracking",
    srcs = ["backtracking.py"],
    deps = [
        ":backtracking_model",
        "//deeplearning/clgen:samplers",
        "//research/cummins_2017_cgo:generative_model",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "backtracking_model",
    srcs = ["backtracking_model.py"],
    deps = [
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/models:tensorflow_backend",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)
