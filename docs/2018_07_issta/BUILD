# Compiler Fuzzing through Deep Learning (ISSTA'18).

exports_files(["README.md"])

genrule(
    name = "2018_07_issta",
    srcs = glob([
        "**/*.sty",
        "**/*.tex",
        "*.sty",
        "acmart.cls",
        "etoolbox.def",
        "img/*.pdf",
        "jpaper.cls",
        "refs.bib",
    ]),
    outs = ["paper.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_issta/paper.tex $@",
    tools = ["//tools:autotex"],
)
