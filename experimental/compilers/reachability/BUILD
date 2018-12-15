# Experiments in learning reachability analysis.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = [
        ":reachability_py_pb2",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "control_flow_graph_test",
    srcs = ["control_flow_graph_test.py"],
    deps = [
        ":control_flow_graph",
        "//third_party/py/absl",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "control_flow_graph_generator",
    srcs = ["control_flow_graph_generator.py"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = [
        ":control_flow_graph",
        "//third_party/py/absl",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "control_flow_graph_generator_test",
    srcs = ["control_flow_graph_generator_test.py"],
    deps = [
        ":control_flow_graph_generator",
        "//third_party/py/absl",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "eval_model",
    srcs = ["eval_model.py"],
    deps = [
        ":reachability_py_pb2",
        ":train_model",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_proto_library(
    name = "reachability_py_pb2",
    protos = ["reachability.proto"],
    visibility = ["//experimental/compilers/reachability:__subpackages__"],
    deps = ["//third_party/py/protobuf"],
)

py_binary(
    name = "train_model",
    srcs = ["train_model.py"],
    deps = [
        ":control_flow_graph",
        ":reachability_py_pb2",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)
