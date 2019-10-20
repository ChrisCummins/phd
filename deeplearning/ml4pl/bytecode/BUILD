# LLVM bytecode datasets.

py_library(
    name = "bytecode_database",
    srcs = ["bytecode_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":bytecode_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "splitters_test",
    srcs = ["splitters_test.py"],
    deps = [
        ":bytecode_database",
        ":splitters",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)
