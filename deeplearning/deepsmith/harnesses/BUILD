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
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":harness",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive:cgen",
        "//gpu/cldrive:data",
        "//gpu/cldrive:driver",
        "//gpu/cldrive:env",
        "//gpu/oclgrind",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//lib/labm8:system",
        "//third_party/py/absl",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "cldrive_test",
    srcs = ["cldrive_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//gpu/oclgrind",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
py3_image(
    name = "cldrive_test_image",
    srcs = ["cldrive_test.py"],
    base = "@base//image",
    main = "cldrive_test.py",
    deps = [
        ":cldrive",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "cl_launcher",
    srcs = ["cl_launcher.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":harness",
        "//compilers/clsmith:cl_launcher",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive:driver",
        "//gpu/cldrive:env",
        "//lib/labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cl_launcher",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//gpu/cldrive:env",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "harness",
    srcs = ["harness.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//lib/labm8:pbutil",
    ],
)
