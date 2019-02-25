# Deprecated cldrive implementation.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

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
    data = ["//gpu/cldrive:native_driver"],
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
    name = "testutil",
    testonly = 1,
    srcs = ["testutil.cc"],
    hdrs = ["testutil.h"],
    deps = [
        ":array_kernel_arg_value",
        ":kernel_arg_value",
        ":scalar_kernel_arg_value",
        "//gpu/cldrive/proto:cldrive_cc_pb2",
        "//third_party/opencl",
    ],
)
