# Library for accessing the global configuration.

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
