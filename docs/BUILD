# Documents and publications.

py_binary(
    name = "minimize_bibtex",
    srcs = ["minimize_bibtex.py"],
    deps = [
        "//labm8:app",
        "//third_party/py/bibtexparser",
    ],
)

py_test(
    name = "minimize_bibtex_test",
    srcs = ["minimize_bibtex_test.py"],
    deps = [
        ":minimize_bibtex",
        "//labm8:test",
    ],
)
