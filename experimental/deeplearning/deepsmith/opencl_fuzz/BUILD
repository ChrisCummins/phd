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
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:generator_pb_py_grpc",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive/legacy:env",
        "//gpu/oclgrind",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:pbutil",
    ],
)

py3_image(
    name = "opencl_fuzz_image",
    srcs = ["opencl_fuzz.py"],
    base = "@phd_base//image",
    main = "opencl_fuzz.py",
    deps = [":opencl_fuzz"],
)

py_test(
    name = "opencl_fuzz_test",
    srcs = ["opencl_fuzz_test.py"],
    deps = [
        ":opencl_fuzz",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "print",
    srcs = ["print.py"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8/py:app",
        "//labm8/py:fmt",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
    ],
)
