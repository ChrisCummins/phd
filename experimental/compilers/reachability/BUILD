# Experiments in learning reachability analysis.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":reachability_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "control_flow_graph_test",
    srcs = ["control_flow_graph_test.py"],
    deps = [
        ":control_flow_graph",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "control_flow_graph_generator",
    srcs = ["control_flow_graph_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_flow_graph",
        "//labm8:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "control_flow_graph_generator_test",
    timeout = "long",
    srcs = ["control_flow_graph_generator_test.py"],
    deps = [
        ":control_flow_graph_generator",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "database",
    srcs = ["database.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":reachability_py_pb2",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

# TODO(cec): Merge into //experimental/compilers/reachability/models:lstm_model
py_binary(
    name = "eval_model",
    srcs = ["eval_model.py"],
    deps = [
        ":reachability_py_pb2",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:pbutil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "llvm_util",
    srcs = ["llvm_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_flow_graph",
        "//compilers/llvm:opt",
        "//labm8:app",
        "//third_party/py/networkx",
        "//third_party/py/pydot",
        "//third_party/py/pyparsing",
    ],
)

py_test(
    name = "llvm_util_test",
    srcs = ["llvm_util_test.py"],
    deps = [
        ":llvm_util",
        "//compilers/llvm:opt",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_proto_library(
    name = "reachability_py_pb2",
    protos = ["reachability.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)
