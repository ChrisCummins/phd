# A Python interface to Oclgrind.
# See: https://github.com/jrprice/Oclgrind

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "oclgrind",
    srcs = ["oclgrind.py"],
    data = select({
        "//:darwin": ["@oclgrind_mac//:oclgrind"],
        "//conditions:default": ["@oclgrind_linux//:oclgrind"],
    }),
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/cldrive:env",
        "//gpu/clinfo/proto:clinfo_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:system",
        requirement("absl-py"),
    ],
)

py_test(
    name = "oclgrind_test",
    srcs = ["oclgrind_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":oclgrind",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
