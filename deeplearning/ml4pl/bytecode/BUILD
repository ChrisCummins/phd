# LLVM bytecode datasets.

py_library(
    name = "bytecode_database",
    srcs = ["bytecode_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//labm8/py:app",
        "//labm8/py:labdate",
        "//labm8/py:pbutil",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "copy_database",
    srcs = ["copy_database.py"],
    deps = [
        ":bytecode_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":bytecode_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "splitters_test",
    srcs = ["splitters_test.py"],
    deps = [
        ":bytecode_database",
        ":splitters",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
