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

py_binary(
    name = "bytecode_db_to_ir_db",
    srcs = ["bytecode_db_to_ir_db.py"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)
