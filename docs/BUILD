genrule(
    name = "2015-08-msc-thesis",
    srcs = glob([
        "2015-08-msc-thesis/**/*.bib",
        "2015-08-msc-thesis/**/*.tex",
        "2015-08-msc-thesis/dat/*",
        "2015-08-msc-thesis/img/*",
        "2015-08-msc-thesis/lib/*",
    ]),
    outs = ["2015-08-msc-thesis.pdf"],
    cmd = "$(location //tools:autotex) docs/2015-08-msc-thesis/thesis.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2015-09-progression-review",
    srcs = glob([
        "2015-09-progression-review/*.tex",
    ]),
    outs = ["2015-09-progression-review.pdf"],
    cmd = "$(location //tools:autotex) docs/2015-09-progression-review/document.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016-01-adapt",
    srcs = glob([
        "2016-01-adapt/tex/**/*.bib",
        "2016-01-adapt/tex/**/*.cls",
        "2016-01-adapt/tex/**/*.tex",
        "2016-01-adapt/tex/img/*",
    ]),
    outs = ["2016-01-adapt.pdf"],
    cmd = "$(location //tools:autotex) docs/2016-01-adapt/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016-01-hlpgpu",
    srcs = glob([
        "2016-01-hlpgpu/tex/**/*.bib",
        "2016-01-hlpgpu/tex/**/*.cls",
        "2016-01-hlpgpu/tex/**/*.tex",
        "2016-01-hlpgpu/tex/img/*",
    ]),
    outs = ["2016-01-hlpgpu.pdf"],
    cmd = "$(location //tools:autotex) docs/2016-01-hlpgpu/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016-06-pldi",
    srcs = glob([
        "2016-06-pldi/abstract.tex",
        "2016-06-pldi/hugh-letter.pdf",
        "2016-06-pldi/preamble.tex",
        "2016-06-pldi/poster.pdf",
    ]),
    outs = ["2016-06-pldi.pdf"],
    cmd = "$(location //tools:autotex) docs/2016-06-pldi/abstract.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016-07-acaces",
    srcs = glob([
        "2016-07-acaces/*.bib",
        "2016-07-acaces/*.bst",
        "2016-07-acaces/*.cls",
        "2016-07-acaces/*.pdf",
        "2016-07-acaces/*.png",
        "2016-07-acaces/*.tex",
    ]),
    outs = ["2016-07-acaces.pdf"],
    cmd = "$(location //tools:autotex) docs/2016-07-acaces/abstract.tex $@",
    tools = ["//tools:autotex"],
)

# FIXME:
# genrule(
#     name = "2016-07-pact",
#     srcs = glob([
#         "2016-07-pact/*.bib",
#         "2016-07-pact/*.bst",
#         "2016-07-pact/*.sty",
#         "2016-07-pact/*.png",
#         "2016-07-pact/*.tex",
#     ]),
#     outs = ["2016-07-pact.pdf"],
#     cmd = "$(location //tools:autotex) docs/2016-07-pact/pact.tex $@",
#     tools = ["//tools:autotex"],
# )

genrule(
    name = "2016-11-first-year-review",
    srcs = glob([
        "2016-11-first-year-review/**/*.tex",
        "2016-11-first-year-review/app/*",
        "2016-11-first-year-review/refs.bib",
    ]),
    outs = ["2016-11-first-year-review.pdf"],
    cmd = "$(location //tools:autotex) docs/2016-11-first-year-review/main.tex $@",
    tools = ["//tools:autotex"],
)

# FIXME: bibliography error
# genrule(
#     name = "2016-12-wip-taco",
#     srcs = glob([
#         "2016-12-wip-taco/*.bst",
#         "2016-12-wip-taco/*.png",
#         "2016-12-wip-taco/*.sty",
#         "2016-12-wip-taco/*.tex",
#         "2016-12-wip-taco/acmsmall*",
#         "2016-12-wip-taco/refs.bib",
#     ]),
#     outs = ["2016-12-wip-taco.pdf"],
#     cmd = "$(location //tools:autotex) docs/2016-12-wip-taco/main.tex $@",
#     tools = ["//tools:autotex"],
# )

genrule(
    name = "2017-02-cgo",
    srcs = glob([
        "2017-02-cgo/tex/**/*.tex",
        "2017-02-cgo/tex/img/*",
        "2017-02-cgo/tex/refs.bib",
        "2017-02-cgo/tex/sigplanconf.cls",
    ]),
    outs = ["2017-02-cgo.pdf"],
    cmd = "$(location //tools:autotex) docs/2017-02-cgo/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2017-09-pact",
    srcs = glob([
        "2017-09-pact/tex/**/*.tex",
        "2017-09-pact/tex/img/*",
        "2017-09-pact/tex/refs.bib",
        "2017-09-pact/tex/IEEEtran.cls",
        "2017-09-pact/tex/IEEEtranS.bst",
    ]),
    outs = ["2017-09-pact.pdf"],
    cmd = "$(location //tools:autotex) docs/2017-09-pact/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)
