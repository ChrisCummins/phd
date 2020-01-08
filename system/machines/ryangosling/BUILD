# Scripts and data for machine ryangosling.

filegroup(
    name = "ryangosling_proto",
    srcs = ["ryangosling.pbtxt"],
)

filegroup(
    name = "ryangosling_zsh",
    srcs = ["ryangosling.zsh"],
)

# This test checks for availability of ryangosling shares. This is incompatible
# with bazel sandboxing and can only be using bazel run, not test.
py_test(
    name = "ryangosling_test",
    srcs = ["ryangosling_test.py"],
    data = [":ryangosling_proto"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
        "//system/machines:machine",
        "//system/machines:mirrored_directory",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        ":ryangosling_proto",
        "//system/machines:machine.par",
    ],
)

py_test(
    name = "samba_shares",
    srcs = ["samba_shares.py"],
    deps = ["//labm8/py:app"],
)
