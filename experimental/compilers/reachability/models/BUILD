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
    name = "lstm_reachability_model",
    srcs = ["lstm_reachability_model.py"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//experimental/compilers/reachability:reachability_py_pb2",
        "//experimental/compilers/reachability:control_flow_graph",
        "//experimental/compilers/reachability/datasets:make_reachability_dataset",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        # Needed to resolve classes in pickled datasets.
        "//experimental/compilers/reachability:llvm_util",  # keep
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:prof",
        "//third_party/py/absl",
        "//labm8:humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lstm_reachability_model_test",
    srcs = ["lstm_reachability_model_test.py"],
    deps = [
        ":lstm_reachability_model",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
