# Post abstract for PLDI'16.

exports_files(["README.txt"])

genrule(
    name = "2016_06_pldi",
    srcs = glob([
        "abstract.tex",
        "hugh_letter.pdf",
        "preamble.tex",
        "poster.pdf",
    ]),
    outs = ["2016_06_pldi.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_06_pldi/abstract.tex $@",
    tools = ["//tools:autotex"],
)
