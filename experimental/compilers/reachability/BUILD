# Experiments in learning reachability analysis.

py_binary(
    name = "eval_model",
    srcs = ["eval_model.py"],
    deps = [
        ":train_model",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "reachability",
    srcs = ["reachability.py"],
    deps = [
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//labm8:fmt",
        "//labm8:fs",
        "//labm8:graph",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "train_model",
    srcs = ["train_model.py"],
    deps = [
        ":reachability",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)
