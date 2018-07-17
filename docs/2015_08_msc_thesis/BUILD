# MSc thesis.

exports_files(["README.txt"])

genrule(
    name = "2015_08_msc_thesis",
    srcs = glob([
        "**/*.bib",
        "**/*.tex",
        "dat/*",
        "img/*",
        "lib/*",
    ]),
    outs = ["2015_08_msc_thesis.pdf"],
    cmd = "$(location //tools:autotex) docs/2015_08_msc_thesis/thesis.tex $@",
    tools = ["//tools:autotex"],
)
