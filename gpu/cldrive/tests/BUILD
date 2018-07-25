load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "testlib",
    srcs = ["testlib.py"],
    deps = [
        "//gpu/cldrive",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

# TODO(cec): Fix me!

# py_test(
#     name = "data_test",
#     size = "small",
#     srcs = ["data_test.py"],
#     default_python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#         ":testlib",
#         "//gpu/cldrive",
#     ],
# )

# py_test(
#     name = "driver_test",
#     size = "small",
#     srcs = ["driver_test.py"],
#     default_python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#         ":testlib",
#         "//gpu/cldrive",
#     ],
# )

# py_test(
#     name = "env_test",
#     size = "small",
#     srcs = ["env_test.py"],
#     default_python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#         ":testlib",
#         "//gpu/cldrive",
#     ],
# )
