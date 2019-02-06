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
    srcs = glob(["rodinia-3.1/**"]),
)

filegroup(
    name = "shoc-1.1.5_files",
    srcs = glob(["shoc-1.1.5/**"]),
)

py_library(
    name = "gpgpu",
    srcs = ["gpgpu.py"],
    data = [
        "@opencl_120_headers//:opencl_headers",
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
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:labdate",
        "//labm8:system",
        "//third_party/py/absl",
        "//gpu/oclgrind",
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
