# Experiments in learning reachability analysis.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "make_training_data",
    srcs = ["make_training_data.py"],
    deps = [
        ":reachability",
        "//experimental/compilers/reachability/proto:reachability_py_pb2",
        requirement("absl-py"),
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
        requirement("absl-py"),
        requirement("numpy"),
    ],
)
