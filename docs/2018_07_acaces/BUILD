# Extended abstract for ACACES'18.

genrule(
    name = "paper",
    srcs = glob([
        "paper.tex",
        "abstract.tex",
        "acaces.bst",
        "acaces.cls",
        "img/clang-crashes.pdf",
        "img/deepsmith.pdf",
        "img/vs-clsmith.pdf",
        "postamble.tex",
        "preamble.tex",
        "refs.bib",
        "sec/conclusion.tex",
        "sec/deepsmith.tex",
        "sec/eval.tex",
        "sec/intro.tex",
    ]),
    outs = ["paper.pdf"],
    cmd = "$(location //tools:autotex) docs/2018_07_acaces/paper.tex $@",
    tools = ["//tools:autotex"],
)
