# Experiemnts in random optimization.

py_library(
    name = "env",
    srcs = ["env.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:llvm_link",
        "//compilers/llvm:opt",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)

py_library(
    name = "envs",
    srcs = ["envs.py"],
    deps = [
        # Note that :env is not directly imported, but required by the gym
        # registration process.
        ":env",
        "//compilers/llvm:opt",
        "//third_party/py/absl",
        "//datasets/benchmarks:bzip2",
        "//datasets/benchmarks:llvm_test_suite",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
    ],
)

py_test(
    name = "envs_test",
    srcs = ["envs_test.py"],
    deps = [
        ":envs",
        "//third_party/py/absl",
        "//third_party/py/gym",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "random_opt",
    srcs = ["random_opt.py"],
    deps = [
        ":env",
        ":envs",
        "//experimental/compilers/random_opt/proto:random_opt_py_pb2",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/gym",
    ],
)
