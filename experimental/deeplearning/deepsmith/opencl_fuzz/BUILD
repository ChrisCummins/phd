# Fuzz testing OpenCL devices using CLgen.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "opencl_fuzz",
    srcs = ["opencl_fuzz.py"],
    deps = [
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/services:cldrive",
        "//deeplearning/deepsmith/services:clgen",
        requirement("absl-py"),
        requirement("humanize"),
    ],
)
