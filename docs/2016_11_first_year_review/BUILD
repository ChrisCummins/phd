# PhD first year review.

exports_files(["README.txt"])

genrule(
    name = "2016_11_first_year_review",
    srcs = glob([
        "**/*.tex",
        "app/*",
        "refs.bib",
    ]),
    outs = ["2016_11_first_year_review.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_11_first_year_review/main.tex $@",
    tools = ["//tools:autotex"],
)
