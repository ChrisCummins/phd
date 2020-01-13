# Learning to use Bazel.

py_test(
    name = "python_binary_test",
    srcs = ["python_binary_test.py"],
    data = [
        ":data_binary",
        ":data_file",
    ],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

filegroup(
    name = "data_file",
    srcs = ["data_file.txt"],
)

cc_binary(
    name = "data_binary",
    srcs = ["data_binary.cc"],
)

py_test(
    name = "stamp_test",
    srcs = ["stamp_test.py"],
    data = [":stamp_file"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

genrule(
    name = "stamp_file",
    outs = ["stamp_file.txt"],
    cmd = """
(
    echo 'Volatile status:';
    cat bazel-out/volatile-status.txt;
    echo
    echo 'Stable status:';
    cat bazel-out/stable-status.txt;
) > $@
""",
    stamp = 1,
)
