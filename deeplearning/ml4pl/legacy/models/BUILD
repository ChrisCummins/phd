# DEPRECATED.
## Models for learning over compiler CFGs.
#
#py_binary(
#    name = "graph_model",
#    srcs = ["graph_model.py"],
#    visibility = ["//visibility:public"],
#    deps = [
#        # Needed to resolve classes in pickled datasets.
#        "//deeplearning/ml4pl:llvm_util",  # keep
#        "//labm8:labdate",
#        "//labm8:prof",
#        "//labm8:app",
#        "//third_party/py/graph_nets",
#        "//third_party/py/networkx",
#        "//third_party/py/numpy",
#        "//third_party/py/pandas",
#        "//third_party/py/sonnet",
#        "//third_party/py/tensorflow",
#    ],
#)
#
#py_test(
#    name = "graph_model_test",
#    srcs = ["graph_model_test.py"],
#    deps = [
#        ":graph_model",
#        "//labm8:app",
#        "//labm8:test",
#        "//third_party/py/pandas",
#        "//third_party/py/pytest",
#    ],
#)
#
#py_binary(
#    name = "lstm_reachability_model",
#    srcs = ["lstm_reachability_model.py"],
#    deps = [
#        "//deeplearning/clgen/corpuses:atomizers",
#        "//deeplearning/ml4pl:ml4pl_pb_py",
#        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
#        "//deeplearning/ml4pl/datasets:make_reachability_dataset",
#        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph_generator",
#        # Needed to resolve classes in pickled datasets.
#        "//deeplearning/ml4pl:llvm_util",  # keep
#        "//labm8:labdate",
#        "//labm8:pbutil",
#        "//labm8:prof",
#        "//labm8:app",
#        "//labm8:humanize",
#        "//third_party/py/keras",
#        "//third_party/py/numpy",
#    ],
#)
#
#py_test(
#    name = "lstm_reachability_model_test",
#    srcs = ["lstm_reachability_model_test.py"],
#    deps = [
#        ":lstm_reachability_model",
#        "//labm8:app",
#        "//labm8:test",
#        "//third_party/py/pytest",
#    ],
#)
