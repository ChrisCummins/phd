load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "opencl_fuzz",
    srcs = ["opencl_fuzz.py"],
    deps = [
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        # TODO(cec): Re-enable CLgen dependency.
        # "//deeplearning/deepsmith/services:clgen",
        "//deeplearning/deepsmith/services:cldrive",
        "//deeplearning/deepsmith/services:randchar",
    ]
)
