py_binary(
    name = "worker_bee",
    srcs = ["worker_bee.py"],
    deps = [
        "//alice:alice_py_pb2",
        "//alice:bazel",
        "//alice:git_repo",
        "//labm8:system",
        "//third_party/py/absl",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "worker_bee_test",
    srcs = ["worker_bee_test.py"],
    deps = [
        ":worker_bee",
        "//labm8:test",
    ],
)
