# A Python interface to Oclgrind.
# See: https://github.com/jrprice/Oclgrind

py_binary(
    name = "oclgrind",
    srcs = ["oclgrind.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//lib/labm8:system", "//lib/labm8:bazelutil"],
    data = select({
        "//:darwin": ["@oclgrind_mac//:oclgrind"],
        "//conditions:default": ["@oclgrind_linux//:oclgrind"],
    })
)

py_test(
    name = "oclgrind_test",
    srcs = ["oclgrind_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [":oclgrind"],
)
