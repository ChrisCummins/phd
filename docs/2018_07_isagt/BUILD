# Position paper for ISAGT'18.

exports_files(["README.md"])

genrule(
    name = "2018_07_isagt",
    srcs = ["paper.tex"],
    outs = ["2018_07_isagt.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_isagt/paper.tex $@",
    tools = ["//tools:autotex"],
)
