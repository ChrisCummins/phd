# Control Flow Graphs.

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "control_flow_graph_test",
    srcs = ["control_flow_graph_test.py"],
    deps = [
        ":control_flow_graph",
        "//labm8/py:app",
        "//labm8/py:test",
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
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
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
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
