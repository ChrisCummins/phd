# Cldrive drives arbitrary OpenCL kernels.

load("@io_bazel_rules_docker//cc:image.bzl", "cc_image")

licenses(["restricted"])  # GPL v3

exports_files([
    "README.md",
    "LICENSE",
])

py_library(
    name = "api",
    srcs = ["api.py"],
    data = [
        ":native_csv_driver",
        ":native_driver",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/cldrive/legacy:env",
        "//gpu/cldrive/proto:cldrive_py_pb2",
        "//gpu/oclgrind",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

cc_binary(
    name = "cldrive",
    srcs = ["cldrive.cc"],
    deps = [
        ":csv_log",
        ":libcldrive",
        "//gpu/clinfo:libclinfo",
        "//phd:app",
        "//phd:logging",
        "@boost//:filesystem",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
    ],
)

cc_image(
    name = "cldrive_image",
    srcs = ["cldrive.cc"],
    base = "@base//image",
    deps = [
        # TODO(cec): This is a duplicate of the dependencies of :cldrive.
        ":csv_log",
        ":libcldrive",
        "//gpu/clinfo:libclinfo",
        "//phd:app",
        "//phd:logging",
        "@boost//:filesystem",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "csv_log",
    srcs = ["csv_log.cc"],
    hdrs = ["csv_log.h"],
    deps = [
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:logging",
        "//phd:port",
        "//phd:string",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "global_memory_arg_value",
    hdrs = ["global_memory_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        "//phd:logging",
        "//phd:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "global_memory_arg_value_test",
    srcs = ["global_memory_arg_value_test.cc"],
    deps = [
        ":global_memory_arg_value",
        ":testutil",
        "//phd:port",
        "//phd:test",
    ],
)

cc_library(
    name = "kernel_arg",
    srcs = ["kernel_arg.cc"],
    hdrs = ["kernel_arg.h"],
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":opencl_type",
        ":opencl_type_util",
        ":opencl_util",
        ":scalar_kernel_arg_value",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:status",
        "//phd:status_macros",
        "//phd:statusor",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_arg_test",
    srcs = ["kernel_arg_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg",
        ":testutil",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:test",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_set",
    srcs = ["kernel_arg_set.cc"],
    hdrs = ["kernel_arg_set.h"],
    deps = [
        ":kernel_arg",
        ":kernel_arg_values_set",
        ":opencl_util",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:logging",
        "//phd:status",
        "//phd:status_macros",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_arg_set_test",
    srcs = ["kernel_arg_set_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":kernel_arg_set",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_value",
    srcs = ["kernel_arg_value.cc"],
    hdrs = ["kernel_arg_value.h"],
    deps = [
        ":opencl_type",
        ":opencl_util",
        ":profiling_data",
        "//phd:logging",
        "//phd:string",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "kernel_arg_values_test",
    srcs = ["kernel_arg_values_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":kernel_arg_value",
        "//phd:port",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_values_set",
    srcs = ["kernel_arg_values_set.cc"],
    hdrs = ["kernel_arg_values_set.h"],
    deps = [
        ":kernel_arg_value",
        "//phd:logging",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "kernel_arg_values_set_test",
    srcs = ["kernel_arg_values_set_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":kernel_arg_values_set",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_driver",
    srcs = ["kernel_driver.cc"],
    hdrs = ["kernel_driver.h"],
    deps = [
        ":kernel_arg_set",
        ":logger",
        ":opencl_util",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//gpu/clinfo:libclinfo",
        "//phd:logging",
        "//phd:status_macros",
        "//phd:statusor",
        "//phd:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_driver_test",
    srcs = ["kernel_driver_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":kernel_driver",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "libcldrive",
    srcs = ["libcldrive.cc"],
    hdrs = ["libcldrive.h"],
    deps = [
        ":kernel_arg_set",
        ":kernel_arg_value",
        ":kernel_arg_values_set",
        ":kernel_driver",
        ":logger",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//gpu/clinfo:libclinfo",
        "//phd:common",
        "//phd:logging",
        "//phd:macros",
        "//phd:statusor",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "libcldrive_test",
    srcs = ["libcldrive_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":libcldrive",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "local_memory_arg_value",
    hdrs = ["local_memory_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        ":profiling_data",
        "//phd:string",
        "//third_party/opencl",
    ],
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":csv_log",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:logging",
        "//phd:status",
    ],
)

cc_binary(
    name = "native_driver",
    srcs = ["native_driver.cc"],
    deps = [
        ":libcldrive",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:pbutil",
    ],
)

cc_binary(
    name = "native_csv_driver",
    srcs = ["native_csv_driver.cc"],
    deps = [
        ":libcldrive",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
    ],
)

cc_library(
    name = "opencl_type",
    srcs = ["opencl_type.cc"],
    hdrs = ["opencl_type.h"],
    deps = [
        "//phd:logging",
        "//phd:status_macros",
        "//phd:statusor",
        "//phd:string",
        "//third_party/opencl",
        "@boost//:variant",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "opencl_type_util",
    srcs = ["opencl_type_util.cc"],
    hdrs = ["opencl_type_util.h"],
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":local_memory_arg_value",
        ":scalar_kernel_arg_value",
        "//third_party/opencl",
    ],
)

cc_library(
    name = "opencl_util",
    srcs = ["opencl_util.cc"],
    hdrs = ["opencl_util.h"],
    deps = [
        ":profiling_data",
        "//phd:logging",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "opencl_util_test",
    srcs = ["opencl_util_test.cc"],
    deps = [
        ":opencl_util",
        ":testutil",
        "//phd:test",
    ],
)

cc_library(
    name = "profiling_data",
    srcs = ["profiling_data.cc"],
    hdrs = ["profiling_data.h"],
    deps = [
        "//phd:port",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "profiling_data_test",
    srcs = ["profiling_data_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":profiling_data",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "scalar_kernel_arg_value",
    srcs = ["scalar_kernel_arg_value.cc"],
    hdrs = ["scalar_kernel_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        "//phd:logging",
        "//phd:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "scalar_kernel_arg_value_test",
    srcs = ["scalar_kernel_arg_value_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":scalar_kernel_arg_value",
        "//phd:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "testutil",
    testonly = 1,
    srcs = ["testutil.cc"],
    hdrs = ["testutil.h"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":scalar_kernel_arg_value",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
