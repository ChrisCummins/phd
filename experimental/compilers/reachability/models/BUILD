# Models for learning over compiler CFGs.

py_binary(
    name = "graph_model",
    srcs = ["graph_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        # Needed to resolve classes in pickled datasets.
        "//experimental/compilers/reachability:llvm_util",  # keep
        "//labm8:labdate",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/graph_nets",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "graph_model_test",
    srcs = ["graph_model_test.py"],
    deps = [
        ":graph_model",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "lstm_model",
    srcs = ["lstm_model.py"],
    deps = [
        "//experimental/compilers/reachability:control_flow_graph",
        "//labm8:prof",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lstm_model_test",
    srcs = ["lstm_model_test.py"],
    deps = [
        ":lstm_model",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
