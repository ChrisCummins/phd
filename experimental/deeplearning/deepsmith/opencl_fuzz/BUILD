# Fuzz testing OpenCL devices using CLgen.

load("@requirements//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "opencl_fuzz",
    srcs = ["opencl_fuzz.py"],
    deps = [
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/services:cldrive",
        "//deeplearning/deepsmith/services:clgen",
        "//gpu/oclgrind",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("humanize"),
    ],
)

py3_image(
    name = "opencl_fuzz_image",
    srcs = ["opencl_fuzz.py"],
    base = "@python3.6//image",
    main = "opencl_fuzz.py",
    deps = [":opencl_fuzz"],
)
