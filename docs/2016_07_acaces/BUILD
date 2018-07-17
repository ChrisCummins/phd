# Post abstract for ACACES'16.

exports_files(["README.txt"])

genrule(
    name = "2016_07_acaces",
    srcs = glob([
        "*.bst",
        "*.cls",
        "*.pdf",
        "*.png",
        "*.tex",
        "refs.bib",
    ]),
    outs = ["2016_07_acaces.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_07_acaces/abstract.tex $@",
    tools = ["//tools:autotex"],
)
