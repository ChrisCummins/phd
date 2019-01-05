# Experiments in learning reachability analysis.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
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
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "control_flow_graph_generator",
    srcs = ["control_flow_graph_generator.py"],
    visibility = ["//visibility:public"],
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
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/networkx",
    ],
)

py_binary(
    name = "cfg_datasets",
    srcs = ["cfg_datasets.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":database",
        ":llvm_util",
        ":reachability_py_pb2",
        "//compilers/llvm:clang",
        "//datasets/linux",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/preprocessors:opencl",
        "//labm8:decorators",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "cfg_datasets_test",
    srcs = ["cfg_datasets_test.py"],
    deps = [
        ":cfg_datasets",
        "//labm8:system",
        "//labm8:test",
        "//third_party/py/absl",
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

py_library(
    name = "llvm_util",
    srcs = ["llvm_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_flow_graph",
        "//compilers/llvm:opt",
        "//third_party/py/absl",
        "//third_party/py/networkx",
        "//third_party/py/pydot",
    ],
)

py_test(
    name = "llvm_util_test",
    srcs = ["llvm_util_test.py"],
    deps = [
        ":llvm_util",
        "//compilers/llvm:opt",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_proto_library(
    name = "reachability_py_pb2",
    protos = ["reachability.proto"],
    visibility = ["//visibility:public"],
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
