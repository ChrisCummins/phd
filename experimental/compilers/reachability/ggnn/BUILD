# Gated graph neural network experiments.

py_binary(
    name = "make_reachability_dataset",
    srcs = ["make_reachability_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//experimental/compilers/reachability:control_and_data_flow_graph",
        "//experimental/compilers/reachability:database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "make_reachability_dataset_test",
    srcs = ["make_reachability_dataset_test.py"],
    deps = [
        "//labm8:test",
        "//labm8:app",
        "//third_party/py/networkx",
        ":make_reachability_dataset",
    ]
)
