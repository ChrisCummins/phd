licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

py_library(
    name = "cldrive",
    srcs = ["__init__.py"],
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
        "@labm8//:err",
    ],
)

py_library(
    name = "driver",
    srcs = ["driver.py"],
    deps = [
        ":args",
        ":env",
        "@labm8//:err",
    ],
)

py_library(
    name = "env",
    srcs = ["env.py"],
)
