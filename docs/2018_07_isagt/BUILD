# Position Paper, ISAGT'18.

exports_files(["README.md"])

genrule(
    name = "paper",
    srcs = ["paper.tex"],
    outs = ["paper.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_isagt/paper.tex $@",
    tools = ["//tools:autotex"],
)
