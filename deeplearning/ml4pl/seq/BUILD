# This moduel defines sequence representations for programs.

py_binary(
    name = "lexers",
    srcs = ["lexers.py"],
    data = [
        "//deeplearning/clgen/corpuses/lexer:lexer_worker",
    ],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lexers_test",
    timeout = "long",
    srcs = ["lexers_test.py"],
    deps = [
        ":lexers",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)
