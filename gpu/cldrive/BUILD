# Cldrive drives arbitrary OpenCL kernels.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

licenses(["restricted"])  # GPL v3

exports_files([
    "README.md",
    "LICENSE",
])

cc_library(
    name = "array_kernel_arg_value",
    hdrs = ["array_kernel_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        "//phd:logging",
        "//phd:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "array_kernel_arg_value_test",
    srcs = ["array_kernel_arg_value_test.cc"],
    deps = [
        ":array_kernel_arg_value",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
)

py_binary(
    name = "cldrive",
    srcs = ["cldrive.py"],
    main = "cldrive.py",
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":cgen",
        ":data",
        ":driver",
        ":env",
        "//third_party/py/numpy",
    ],
)

# py3_image(
#     name = "cldrive_image",
#     srcs = ["cldrive.py"],
#     base = "@base//image",
#     main = "cldrive.py",
#     deps = [":cldrive"],
# )

py_library(
    name = "args",
    srcs = ["args.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pycparser",
    ],
)

py_test(
    name = "args_test",
    size = "small",
    srcs = ["args_test.py"],
    deps = [
        ":args",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "cgen",
    srcs = ["cgen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":driver",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":driver",
        "//labm8:err",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.py"],
    deps = [
        ":data",
        ":testlib",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_binary(
    name = "driver",
    srcs = ["driver.py"],
    data = [":native_driver"],
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":env",
        "//gpu/cldrive/proto:cldrive_py_pb2",
        "//labm8:bazelutil",
        "//labm8:err",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.py"],
    deps = [
        ":data",
        ":driver",
        ":testlib",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "env",
    srcs = ["env.py"],
    data = ["//gpu/clinfo"],
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_py_pb2",
        "//gpu/oclgrind",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.py"],
    deps = [
        ":env",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

cc_library(
    name = "kernel_arg",
    srcs = ["kernel_arg.cc"],
    hdrs = ["kernel_arg.h"],
    deps = [
        ":array_kernel_arg_value",
        ":kernel_arg_value",
        ":scalar_kernel_arg_value",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//phd:status",
        "//phd:statusor",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_arg_test",
    srcs = ["kernel_arg_test.cc"],
    deps = [
        ":kernel_arg",
        "//phd:test",
        ":array_kernel_arg_value",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//third_party/opencl",
        "//third_party/opencl:libOpenCL.so",
    ],
)

cc_library(
    name = "kernel_arg_set",
    srcs = ["kernel_arg_set.cc"],
    hdrs = ["kernel_arg_set.h"],
    deps = [
        ":kernel_arg",
        ":kernel_arg_values_set",
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
    deps = [
        ":kernel_arg_set",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
)

cc_library(
    name = "kernel_driver",
    srcs = ["kernel_driver.cc"],
    hdrs = ["kernel_driver.h"],
    deps = [
        ":kernel_arg_set",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
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
    deps = [
        ":kernel_driver",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
)

cc_library(
    name = "kernel_arg_value",
    srcs = ["kernel_arg_value.cc"],
    hdrs = ["kernel_arg_value.h"],
    deps = [
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
    deps = [
        ":kernel_arg_value",
        "//phd:port",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
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
    deps = [
        ":kernel_arg_values_set",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
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
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//gpu/clinfo:libclinfo",
        "//phd:common",
        "//phd:logging",
        "//phd:macros",
        "//phd:statusor",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "libcldrive_test",
    srcs = ["libcldrive_test.cc"],
    deps = [
        ":libcldrive",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
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
    deps = [
        ":scalar_kernel_arg_value",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
)

py_library(
    name = "testlib",
    testonly = 1,
    srcs = ["testlib.py"],
    deps = [
        "//third_party/py/numpy",
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

cc_library(
    name = "opencl_util",
    hdrs = ["opencl_util.h"],
    deps = [
        ":profiling_data",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "opencl_util_test",
    srcs = ["opencl_util_test.cc"],
    deps = [
        ":opencl_util",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
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
    deps = [
        ":profiling_data",
        "//phd:test",
        "//third_party/opencl:libOpenCL.so",
    ],
)
