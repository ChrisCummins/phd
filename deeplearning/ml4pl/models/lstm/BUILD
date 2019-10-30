py_library(
    name = "bytecode2seq",
    srcs = ["bytecode2seq.py"],
    data = ["//deeplearning/clgen/corpuses/lexer:lexer_worker"],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "derive_vocabulary",
    srcs = ["derive_vocabulary.py"],
    deps = [
        ":bytecode2seq",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//labm8:sqlutil",
    ],
)
