genrule(
    name = "2015_08_msc_thesis",
    srcs = glob([
        "2015_08_msc_thesis/**/*.bib",
        "2015_08_msc_thesis/**/*.tex",
        "2015_08_msc_thesis/dat/*",
        "2015_08_msc_thesis/img/*",
        "2015_08_msc_thesis/lib/*",
    ]),
    outs = ["2015_08_msc_thesis.pdf"],
    cmd = "$(location //tools:autotex) docs/2015_08_msc_thesis/thesis.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2015_09_progression_review",
    srcs = glob([
        "2015_09_progression_review/*.tex",
    ]),
    outs = ["2015_09_progression_review.pdf"],
    cmd = "$(location //tools:autotex) docs/2015_09_progression_review/document.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016_01_adapt",
    srcs = glob([
        "2016_01_adapt/tex/**/*.bib",
        "2016_01_adapt/tex/**/*.cls",
        "2016_01_adapt/tex/**/*.tex",
        "2016_01_adapt/tex/img/*",
    ]),
    outs = ["2016_01_adapt.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_01_adapt/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016_01_hlpgpu",
    srcs = glob([
        "2016_01_hlpgpu/tex/**/*.bib",
        "2016_01_hlpgpu/tex/**/*.cls",
        "2016_01_hlpgpu/tex/**/*.tex",
        "2016_01_hlpgpu/tex/img/*",
    ]),
    outs = ["2016_01_hlpgpu.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_01_hlpgpu/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016_06_pldi",
    srcs = glob([
        "2016_06_pldi/abstract.tex",
        "2016_06_pldi/hugh_letter.pdf",
        "2016_06_pldi/preamble.tex",
        "2016_06_pldi/poster.pdf",
    ]),
    outs = ["2016_06_pldi.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_06_pldi/abstract.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2016_07_acaces",
    srcs = glob([
        "2016_07_acaces/*.bib",
        "2016_07_acaces/*.bst",
        "2016_07_acaces/*.cls",
        "2016_07_acaces/*.pdf",
        "2016_07_acaces/*.png",
        "2016_07_acaces/*.tex",
    ]),
    outs = ["2016_07_acaces.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_07_acaces/abstract.tex $@",
    tools = ["//tools:autotex"],
)

# TODO(cec):
# genrule(
#     name = "2016_07_pact",
#     srcs = glob([
#         "2016_07_pact/*.bib",
#         "2016_07_pact/*.bst",
#         "2016_07_pact/*.sty",
#         "2016_07_pact/*.png",
#         "2016_07_pact/*.tex",
#     ]),
#     outs = ["2016_07_pact.pdf"],
#     cmd = "$(location //tools:autotex) docs/2016_07_pact/pact.tex $@",
#     tools = ["//tools:autotex"],
# )

genrule(
    name = "2016_11_first_year_review",
    srcs = glob([
        "2016_11_first_year_review/**/*.tex",
        "2016_11_first_year_review/app/*",
        "2016_11_first_year_review/refs.bib",
    ]),
    outs = ["2016_11_first_year_review.pdf"],
    cmd = "$(location //tools:autotex) docs/2016_11_first_year_review/main.tex $@",
    tools = ["//tools:autotex"],
)

# TODO(cec): bibliography error
# genrule(
#     name = "2016_12_wip_taco",
#     srcs = glob([
#         "2016_12_wip_taco/*.bst",
#         "2016_12_wip_taco/*.png",
#         "2016_12_wip_taco/*.sty",
#         "2016_12_wip_taco/*.tex",
#         "2016_12_wip_taco/acmsmall*",
#         "2016_12_wip_taco/refs.bib",
#     ]),
#     outs = ["2016_12_wip_taco.pdf"],
#     cmd = "$(location //tools:autotex) docs/2016_12_wip_taco/main.tex $@",
#     tools = ["//tools:autotex"],
# )

genrule(
    name = "2017_02_cgo",
    srcs = glob([
        "2017_02_cgo/tex/**/*.tex",
        "2017_02_cgo/tex/img/*",
        "2017_02_cgo/tex/refs.bib",
        "2017_02_cgo/tex/sigplanconf.cls",
    ]),
    outs = ["2017_02_cgo.pdf"],
    cmd = "$(location //tools:autotex) docs/2017_02_cgo/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "2017_09_pact",
    srcs = glob([
        "2017_09_pact/tex/**/*.tex",
        "2017_09_pact/tex/img/*",
        "2017_09_pact/tex/refs.bib",
        "2017_09_pact/tex/IEEEtran.cls",
        "2017_09_pact/tex/IEEEtranS.bst",
    ]),
    outs = ["2017_09_pact.pdf"],
    cmd = "$(location //tools:autotex) docs/2017_09_pact/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)

genrule(
    name = "cv",
    srcs = glob([
        "cv/**/*.tex",
    ]),
    outs = ["cv.pdf"],
    cmd = "$(location //tools:autotex) docs/cv/cv.tex $@",
    tools = ["//tools:autotex"],
)
