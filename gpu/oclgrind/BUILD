# A Python interface to Oclgrind.
# See: https://github.com/jrprice/Oclgrind

py_binary(
    name = "oclgrind",
    srcs = ["oclgrind.py"],
    data = select({
        "//:darwin": ["@oclgrind_mac//:oclgrind"],
        "//conditions:default": ["@oclgrind_linux//:oclgrind"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_py_pb2",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "oclgrind_test",
    srcs = ["oclgrind_test.py"],
    data = ["//gpu/oclgrind/test/data:oclgrind_working"],
    deps = [
        ":oclgrind",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)
