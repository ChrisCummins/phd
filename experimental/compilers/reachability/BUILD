# Experiments in learning reachability analysis.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "reachability",
    srcs = ["reachability.py"],
    deps = [
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        "//lib/labm8:fmt",
        "//lib/labm8:fs",
        "//lib/labm8:graph",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "train_model",
    srcs = ["train_model.py"],
    deps = [
        ":reachability",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen:telemetry",
        "//lib/labm8:pbutil",
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        requirement("absl-py"),
        requirement("keras"),
        requirement("numpy"),
        requirement("humanize"),
    ],
)
