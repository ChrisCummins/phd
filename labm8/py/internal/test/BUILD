load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "print_workspace_status",
    testonly = 1,
    srcs = ["print_workspace_status.py"],
    data = ["//labm8/py/internal:workspace_status"],
)

py_test(
    name = "workspace_status_par_test",
    srcs = ["workspace_status_par_test.py"],
    data = [":print_workspace_status.par"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

py_test(
    name = "workspace_status_test",
    srcs = ["workspace_status_test.py"],
    data = ["//labm8/py/internal:workspace_status"],
    deps = [
        "//labm8/py:test",
    ],
)
