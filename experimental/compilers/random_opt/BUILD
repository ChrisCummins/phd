# Experiemnts in random optimization.

py_library(
    name = "implementation",
    srcs = ["implementation.py"],
    deps = [
        "//compilers/llvm",
        "//compilers/llvm:clang",
        "//compilers/llvm:llvm_link",
        "//compilers/llvm:opt",
        "//experimental/compilers/random_opt/proto:random_opt_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:jsonutil",
        "//labm8/py:labdate",
        "//labm8/py:pbutil",
        "//labm8/py:text",
        "//third_party/py/gym",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "implementation_test",
    srcs = ["implementation_test.py"],
    deps = [
        ":implementation",
        "//compilers/llvm:clang",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "environments",
    srcs = ["environments.py"],
    deps = [
        # Note that :implementation is not directly imported, but required by
        # the gym registration process.
        ":implementation",
        "//labm8/py:labtypes",
        "//labm8/py:app",
        "//datasets/benchmarks:bzip2",
        "//datasets/benchmarks:llvm_test_suite",
        "//experimental/compilers/random_opt/proto:random_opt_pb_py",
    ],
)

py_test(
    name = "environments_test",
    srcs = ["environments_test.py"],
    deps = [
        ":environments",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/gym",
    ],
)

py_binary(
    name = "results_to_dot",
    srcs = ["results_to_dot.py"],
    deps = [
        "//experimental/compilers/random_opt/proto:random_opt_pb_py",
        "//labm8/py:app",
        "//labm8/py:graph",
        "//labm8/py:pbutil",
    ],
)

py_binary(
    name = "random_opt",
    srcs = ["random_opt.py"],
    deps = [
        ":environments",
        ":implementation",
        "//experimental/compilers/random_opt/proto:random_opt_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//third_party/py/gym",
    ],
)
