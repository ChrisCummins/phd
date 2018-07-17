# My CV / Resume.

genrule(
    name = "cv",
    srcs = glob([
        "cv.tex",
        "includes/*.tex",
        "sec/*.tex",
    ]),
    outs = ["cv.pdf"],
    cmd = "$(location //tools:autotex) docs/cv/cv.tex $@",
    tools = ["//tools:autotex"],
)
