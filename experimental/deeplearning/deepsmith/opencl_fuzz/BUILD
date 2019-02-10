# Fuzz testing OpenCL devices using CLgen.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "opencl_fuzz",
    srcs = ["opencl_fuzz.py"],
    data = [
        "@CLSmith//:cl_launcher_src",
    ],
    deps = [
        "//deeplearning/deepsmith/difftests",
        "//deeplearning/deepsmith/difftests:opencl",
        "//deeplearning/deepsmith/generators:clgen_pretrained",
        "//deeplearning/deepsmith/generators:clsmith",
        "//deeplearning/deepsmith/harnesses:cl_launcher",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive:env",
        "//gpu/oclgrind",
        "//labm8:bazelutil",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
    ],
)

py3_image(
    name = "opencl_fuzz_image",
    srcs = ["opencl_fuzz.py"],
    base = "@base//image",
    main = "opencl_fuzz.py",
    deps = [":opencl_fuzz"],
)

py_test(
    name = "opencl_fuzz_test",
    srcs = ["opencl_fuzz_test.py"],
    deps = [
        ":opencl_fuzz",
        "//labm8:pbutil",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "print",
    srcs = ["print.py"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//labm8:fmt",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
    ],
)
