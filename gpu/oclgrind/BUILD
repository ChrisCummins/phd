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
        "//labm8:bazelutil",
        "//labm8:system",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "oclgrind_test",
    srcs = ["oclgrind_test.py"],
    deps = [
        ":oclgrind",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
