py_binary(
    name = "ledger",
    srcs = ["ledger.py"],
    deps = [
        "//alice:alice_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "ledger_test",
    srcs = ["ledger_test.py"],
    deps = [
        ":ledger",
        "//labm8:test",
        "//third_party/py/grpcio",
    ],
)
