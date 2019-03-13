# Testcase harnes implementations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "cldrive",
    srcs = ["cldrive.py"],
    data = ["@opencl_120_headers//:opencl_headers"] + select({
        "//:darwin": [
            "@llvm_mac//:clang",
            "@llvm_mac//:libcxx_headers",
            "@llvm_mac//:libcxx_libs",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
            "@llvm_linux//:libcxx_headers",
            "@llvm_linux//:libcxx_libs",
            "@libopencl//:libOpenCL.so",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":harness",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive/legacy:cgen",
        "//gpu/cldrive/legacy:data",
        "//gpu/cldrive/legacy:driver",
        "//gpu/cldrive/legacy:env",
        "//gpu/oclgrind",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:system",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "cldrive_test",
    srcs = ["cldrive_test.py"],
    deps = [
        ":cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//gpu/cldrive/legacy:env",
        "//gpu/oclgrind",
        "//labm8:app",
        "//labm8:test",
    ],
)

py3_image(
    name = "cldrive_image",
    srcs = ["cldrive.py"],
    base = "@base//image",
    main = "cldrive.py",
    deps = [":cldrive"],
)

# A docker image for the cldrive unit tests. This is useful for testing the
# hemetic-ness (definitely not a word) of the cldrive harness.
# FIXME(cec): //labm8:test is marked test-only so cannot be depended on.
# py3_image(
#     name = "cldrive_test_image",
#     srcs = ["cldrive_test.py"],
#     base = "@base//image",
#     main = "cldrive_test.py",
#     deps = [
#         ":cldrive",
#         "//labm8:test",
#         "//labm8:app",
#     ],
# )

py_binary(
    name = "cl_launcher",
    srcs = ["cl_launcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":harness",
        "//compilers/clsmith:cl_launcher",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive/legacy:driver",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:labdate",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    deps = [
        ":cl_launcher",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//labm8:pbutil",
    ],
)
