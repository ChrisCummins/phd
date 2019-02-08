# GPGPU benchmark suites.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

filegroup(
    name = "amd-app-sdk-3.0_files",
    srcs = glob(["amd-app-sdk-3.0/**"]),
)

filegroup(
    name = "npb-3.3_files",
    srcs = glob(["npb-3.3/**"]),
)

filegroup(
    name = "nvidia-4.2_files",
    srcs = glob(["nvidia-4.2/**"]),
)

filegroup(
    name = "parboil-0.2_files",
    srcs = glob(["parboil-0.2/**"]),
)

filegroup(
    name = "polybench-gpu-1.0_files",
    srcs = glob(["polybench-gpu-1.0/**"]),
)

filegroup(
    name = "rodinia-3.1_files",
    srcs = glob(["rodinia-3.1/**"]) + [
        "@rodinia_data//:data_files",
    ],
)

filegroup(
    name = "shoc-1.1.5_files",
    srcs = glob(["shoc-1.1.5/**"]),
)

py_proto_library(
    name = "gpgpu_py_pb2",
    protos = ["gpgpu.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_binary(
    name = "gpgpu",
    srcs = ["gpgpu.py"],
    data = [
        "@opencl_120_headers//:opencl_headers",
        "//datasets/benchmarks/gpgpu/dummy_just_for_testing:dummy_benchmark",
        ":amd-app-sdk-3.0_files",
        ":npb-3.3_files",
        ":nvidia-4.2_files",
        ":parboil-0.2_files",
        ":polybench-gpu-1.0_files",
        ":rodinia-3.1_files",
        ":shoc-1.1.5_files",
        "//gpu/libcecl:libcecl.so",
        "//gpu/libcecl:libcecl_header",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL.so"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gpgpu_py_pb2",
        "//gpu/oclgrind",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:system",
        "//labm8:text",
        "//third_party/py/absl",
        "//third_party/py/humanize",
    ],
)

py_test(
    name = "gpgpu_test",
    srcs = ["gpgpu_test.py"],
    deps = [
        ":gpgpu",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

sh_test(
    name = "gpgpu_smoke_test",
    srcs = ["gpgpu_smoke_test.sh"],
    data = [":gpgpu"],
)
