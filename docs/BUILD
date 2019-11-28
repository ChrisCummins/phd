# Documents and publications.

filegroup(
    name = "acronyms",
    srcs = ["acronyms.json"],
)

py_binary(
    name = "deacronym_bibtex",
    srcs = ["deacronym_bibtex.py"],
    data = [":acronyms"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:jsonutil",
        "//third_party/py/bibtexparser",
    ],
)

py_binary(
    name = "minimize_bibtex",
    srcs = ["minimize_bibtex.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/bibtexparser",
    ],
)

py_test(
    name = "minimize_bibtex_test",
    srcs = ["minimize_bibtex_test.py"],
    deps = [
        ":minimize_bibtex",
        "//labm8/py:test",
    ],
)
