# Extended abstract for ACACES'18.

genrule(
    name = "paper",
    srcs = glob([
        "2018_07_acaces/refs.bib",
        "2018_07_acaces/**/*.tex",
        "2018_07_acaces/img/*.pdf",
        "2018_07_acaces/acaces.bst",
        "2018_07_acaces/acaces.cls",
    ]),
    outs = ["paper.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_acaces/thesis.tex $@",
    tools = ["//tools:autotex"],
)
