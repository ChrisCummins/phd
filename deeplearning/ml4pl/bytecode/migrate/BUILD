# Scripts for data migrations.

py_binary(
    name = "add_bytecode_sha1_column",
    srcs = ["add_bytecode_sha1_column.py"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:prof",
    ],
)

py_binary(
    name = "add_missing_devmap",
    srcs = ["add_missing_devmap.py"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
    ],
)

py_binary(
    name = "mark_duplicates",
    srcs = ["mark_duplicates.py"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8/py:app",
    ],
)
