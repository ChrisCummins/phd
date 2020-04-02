# Scripts and data for machine ryangosling.

filegroup(
    name = "bodie_configs",
    srcs = [
        "bodie.pbtxt",
        "bodie.zsh",
    ],
)

# This test checks for availability of ryangosling shares. This is incompatible
# with bazel sandboxing and can only use bazel run, not test.
py_test(
    name = "ryangosling_test",
    srcs = ["ryangosling_test.py"],
    data = [":install"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//system/machines:machine",
        "//system/machines:mirrored_directory",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        ":bodie_configs",
        "//system/machines:machine.par",
    ],
)

py_test(
    name = "samba_shares",
    srcs = ["samba_shares.py"],
    deps = ["//labm8/py:app"],
)
