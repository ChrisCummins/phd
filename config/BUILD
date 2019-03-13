# Library for accessing the global configuration.

py_library(
    name = "build_info",
    srcs = ["build_info.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":getconfig",
        "//labm8:app",
        "//third_party/py/git",
    ],
)

py_test(
    name = "build_info_test",
    srcs = ["build_info_test.py"],
    deps = [
        ":build_info",
        "//labm8:test",
    ],
)

py_library(
    name = "getconfig",
    srcs = ["getconfig.py"],
    data = ["//:config"],
    visibility = ["//visibility:public"],
    deps = [
        "//config/proto:config",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "getconfig_test",
    srcs = ["getconfig_test.py"],
    deps = [
        ":getconfig",
        "//labm8:app",
        "//labm8:test",
    ],
)
