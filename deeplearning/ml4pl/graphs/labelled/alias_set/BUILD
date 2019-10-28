# Annotating graphs with alias sets.

py_library(
    name = "alias_set",
    srcs = ["alias_set.py"],
)

py_binary(
    name = "make_alias_set_dataset",
    srcs = ["make_alias_set_dataset.py"],
    deps = [
        ":alias_set",
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
        "//labm8:fs",
    ],
)
