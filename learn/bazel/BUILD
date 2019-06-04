# Learning to use Bazel.

py_test(
    name = "python_binary_test",
    srcs = ["python_binary_test.py"],
    data = [
        ":data_binary",
        ":data_file",
    ],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
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
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:test",
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

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/tensorflow",
    ],
)
