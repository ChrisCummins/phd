# Gated graph neural network experiments.

py_binary(
    name = "make_reachability_dataset",
    srcs= ["make_reachability_dataset.py"],
    deps = [
        "//third_party/py/networkx",
        "//experimental/compilers/reachability:control_and_data_flow_graph",
        "//experimental/compilers/reachability:database",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
    ],
    visibility = ["//visibility:public"],
)
