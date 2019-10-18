# Control Flow Graphs.

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:ml4pl_pb_py",
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
    name = "llvm_util",
    srcs = ["llvm_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_flow_graph",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//labm8:app",
        "//labm8:pbutil",
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
