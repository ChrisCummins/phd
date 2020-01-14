# This package defines a py_binary target for my fork of black.
# github.com/ChrisCummins/black-2spaces

py_binary(
    name = "black",
    srcs = [
        "black.py",
    ] + glob(["blib2to3/**/*.py"]),
    data = [
        "blib2to3/Grammar.txt",
        "blib2to3/PatternGrammar.txt",
    ],
    main = "black.py",
    visibility = ["//visibility:public"],
    deps = [
        "@phd//third_party/py/appdirs",
        "@phd//third_party/py/attrs",
        "@phd//third_party/py/click",
        "@phd//third_party/py/dataclasses",
        "@phd//third_party/py/mypy_extensions",
        "@phd//third_party/py/pathspec",
        "@phd//third_party/py/regex",
        "@phd//third_party/py/toml",
        "@phd//third_party/py/typed_ast",
        "@phd//third_party/py/typing_extensions",
    ],
)
