# DeepSmith: Compiler Fuzzing through Deep Learning (ACACES'18).

exports_files(["README.md"])

genrule(
    name = "2018_07_acaces",
    srcs = glob([
        "**/*.tex",
        "acaces.bst",
        "acaces.cls",
        "img/*.pdf",
        "refs.bib",
    ]),
    outs = ["paper.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_acaces/paper.tex $@",
    tools = ["//tools:autotex"],
)
