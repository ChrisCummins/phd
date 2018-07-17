# MSc progression review.

exports_files(["README.txt"])

genrule(
    name = "2015_09_progression_review",
    srcs = glob([
        "*.tex",
    ]),
    outs = ["2015_09_progression_review.pdf"],
    cmd = "$(location //tools:autotex) docs/2015_09_progression_review/document.tex $@",
    tools = ["//tools:autotex"],
)
