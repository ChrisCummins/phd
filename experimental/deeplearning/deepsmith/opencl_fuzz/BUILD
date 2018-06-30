# Fuzz testing OpenCL devices using CLgen.

load("@requirements//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "opencl_fuzz",
    srcs = ["opencl_fuzz.py"],
    deps = [
        "//deeplearning/deepsmith/generators:clgen",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/oclgrind",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("humanize"),
    ],
)

py_test(
    name = "opencl_fuzz_test",
    srcs = ["opencl_fuzz_test.py"],
    deps = [
        ":opencl_fuzz",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py3_image(
    name = "opencl_fuzz_image",
    srcs = ["opencl_fuzz.py"],
    base = "@python3.6//image",
    main = "opencl_fuzz.py",
    deps = [":opencl_fuzz"],
)

py_binary(
    name = "print",
    srcs = ["print.py"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//lib/labm8:fmt",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("humanize"),
    ],
)
