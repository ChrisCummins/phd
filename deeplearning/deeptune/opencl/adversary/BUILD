# Adverserial counter-examples for sequential modeling of source code.

py_binary(
    name = "adversary",
    srcs = ["adversary.py"],
    deps = [
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8:prof",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "opencl_deadcode_inserter",
    srcs = ["opencl_deadcode_inserter.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//gpu/cldrive/legacy:args",
        "//labm8:fmt",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "opencl_deadcode_inserter_test",
    srcs = ["opencl_deadcode_inserter_test.py"],
    deps = [
        ":opencl_deadcode_inserter",
        "//compilers/llvm:clang",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "opencl_deadcode_inserter_fuzz_test",
    size = "enormous",
    srcs = ["opencl_deadcode_inserter_fuzz_test.py"],
    deps = [
        ":opencl_deadcode_inserter",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//gpu/oclgrind",
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)
