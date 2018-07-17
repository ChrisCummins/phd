# Rejected submission for PACT'16 SRC.

exports_files(["README.txt"])

genrule(
   name = "2016_07_pact",
   srcs = glob([
       "refs.bib",
       "latex8.bst",
       "*.sty",
       "*.png",
       "*.tex",
   ]),
   outs = ["2016_07_pact.pdf"],
   cmd = "$(location //tools:autotex) docs/2016_07_pact/pact.tex $@",
   tools = ["//tools:autotex"],
)
