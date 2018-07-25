# Cldrive drives arbitrary OpenCL kernels.

load("@requirements//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_binary(
    name = "cldrive",
    srcs = ["cldrive.py"],
    default_python_version = "PY3",
    main = "cldrive.py",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":cgen",
        ":data",
        ":driver",
        ":env",
        requirement("numpy"),
    ],
)

py3_image(
    name = "cldrive_image",
    srcs = ["cldrive.py"],
    base = "@base//image",
    main = "cldrive.py",
    deps = [":cldrive"],
)

py_library(
    name = "args",
    srcs = ["args.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        requirement("pycparser"),
        requirement("pycparserext"),
    ],
)

py_test(
    name = "args_test",
    size = "small",
    srcs = ["args_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":args",
        requirement("pytest"),
        requirement("absl-py"),
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
        "//lib/labm8:err",
        requirement("numpy"),
    ],
)

py_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data",
        ":testlib",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "driver",
    srcs = ["driver.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":env",
        "//lib/labm8:err",
        requirement("numpy"),
    ],
)

py_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data",
        ":driver",
        ":testlib",
        requirement("absl-py"),
        requirement("numpy"),
        requirement("pytest"),
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
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":env",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "testlib",
    testonly = 1,
    srcs = ["testlib.py"],
    deps = [
        requirement("numpy"),
    ],
)
