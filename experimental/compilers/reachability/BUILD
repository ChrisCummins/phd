# Experiments in learning reachability analysis.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":random_opt_pb_py",
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
        ":random_opt_pb_py",
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
        ":random_opt_pb_py",
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

proto_library(
    name = "reachability_pb",
    srcs = ["reachability.proto"],
)

python_proto_library(
    name = "random_opt_pb_py",
    visibility = ["//visibility:public"],
    deps = [":reachability_pb"],
)
