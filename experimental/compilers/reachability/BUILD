# Experiments in learning reachability analysis.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "eval_model",
    srcs = ["eval_model.py"],
    deps = [
        ":reachability",
        ":train_model",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("keras"),
        "//third_party/py/numpy",
        requirement("humanize"),
    ],
)

py_binary(
    name = "reachability",
    srcs = ["reachability.py"],
    deps = [
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//lib/labm8:fmt",
        "//lib/labm8:fs",
        "//lib/labm8:graph",
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
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("keras"),
        "//third_party/py/numpy",
        requirement("humanize"),
    ],
)
