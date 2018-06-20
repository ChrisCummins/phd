load("@requirements//:requirements.bzl", "requirement")

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "cldrive",
    srcs = [
        "__init__.py",
        "cldrive.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":args",
        ":cgen",
        ":data",
        ":driver",
        ":env",
    ],
)

py_library(
    name = "args",
    srcs = ["args.py"],
    deps = [
        requirement("numpy"),
        requirement("pycparser"),
        requirement("pycparserext"),
    ],
)

py_library(
    name = "cgen",
    srcs = ["cgen.py"],
    deps = [
        ":args",
        ":driver",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        ":args",
        ":driver",
        "//lib/labm8:err",
        requirement("numpy"),
    ],
)

py_library(
    name = "driver",
    srcs = ["driver.py"],
    deps = [
        ":args",
        ":env",
        "//lib/labm8:err",
        requirement("numpy"),
    ],
)

py_library(
    name = "env",
    srcs = ["env.py"],
    visibility = ["//gpu:__subpackages__"],
    deps = [
        requirement("pyopencl"),
        requirement("pytools"),
        requirement("absl-py"),
    ],
)
